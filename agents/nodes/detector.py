"""Detector agent: fundus image -> DR grade, confidence and safety flags.

Pipeline (matches the training notebook):
    image -> resize 256x256 -> median blur -> CLAHE on LAB L channel -> RGB
          -> frozen ResNet50 + InceptionV3 (2048 + 2048 features)
          -> StandardScaler -> RBF SVM -> five probabilities

Bundle schema (``dr_svm_bundle.joblib``, a dict)::

    {
        "scaler": fitted StandardScaler or None,
        "svm": fitted SVC(probability=True),      # alias: "model" / "classifier"
        "t_severe": float,                        # optional, falls back to settings
        "t_referable": float,                     # optional, falls back to settings
    }

Grade escalation:
    urgent      = P(Severe) + P(Proliferative) >= t_severe
    referable   = P(Moderate) + P(Severe) + P(Proliferative) >= t_referable
    final grade = max(raw grade, 3 if urgent else 2 if referable else raw grade)

The heavy ML imports (cv2, numpy, tensorflow, joblib) are lazy so the scoring
logic can be tested without them.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from agents.config import settings
from agents.state import (
    DR_State,
    GRADE_LABELS,
    REFERABLE_GRADE,
    SEVERE_GRADE,
)

IMAGE_SIZE = 256
_MODEL_KEYS = ("svm", "model", "classifier")
_SCALER_KEYS = ("scaler", "standard_scaler")
_T_SEVERE_KEYS = ("t_severe", "T_SEVERE", "threshold_severe")
_T_REFERABLE_KEYS = ("t_referable", "T_REFERABLE", "threshold_referable")


# ---------------------------------------------------------------------------
# Bundle loading
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _load_bundle_cached(path: str):
    import joblib

    return joblib.load(path)


def load_bundle(path: str | Path | None = None):
    """Load the joblib bundle from ``DR_MODEL_PATH`` (or an explicit path)."""

    bundle_path = Path(path) if path is not None else settings.model_path
    if not bundle_path.is_file():
        raise FileNotFoundError(
            f"model bundle not found: {bundle_path} "
            "(train the notebook or set DR_MODEL_PATH)"
        )
    return _load_bundle_cached(str(bundle_path.expanduser().resolve()))


def _lookup(bundle, keys: tuple[str, ...], what: str):
    """Find a value in a dict-like or attribute-based bundle."""

    if isinstance(bundle, dict):
        for key in keys:
            if key in bundle:
                return bundle[key]
    for key in keys:
        if hasattr(bundle, key):
            return getattr(bundle, key)
    raise KeyError(f"model bundle has no {what} (looked for {keys})")


def _optional_threshold(bundle, keys: tuple[str, ...], fallback: float) -> float:
    try:
        return float(_lookup(bundle, keys, keys[0]))
    except KeyError:
        return float(fallback)


# ---------------------------------------------------------------------------
# Image preprocessing and feature extraction
# ---------------------------------------------------------------------------


def preprocess_image(image_path: str):
    """Return the model-ready RGB image: 256x256, median blur, CLAHE."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "opencv-python is required for the detector; "
            "pip install opencv-python-headless"
        ) from exc

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"could not read image: {image_path}")

    image = cv2.resize(
        image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA
    )
    image = cv2.medianBlur(image, 3)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab = cv2.merge((clahe.apply(l_channel), a_channel, b_channel))

    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


@lru_cache(maxsize=1)
def _get_feature_models():
    """Build the frozen ResNet50 and InceptionV3 feature extractors once."""

    try:
        import numpy as np
        from tensorflow.keras.applications import InceptionV3, ResNet50
        from tensorflow.keras.applications.inception_v3 import (
            preprocess_input as inception_preprocess,
        )
        from tensorflow.keras.applications.resnet50 import (
            preprocess_input as resnet_preprocess,
        )
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "tensorflow and numpy are required for feature extraction; "
            "use Python 3.11/3.12 and pip install tensorflow numpy"
        ) from exc

    resnet = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        pooling="avg",
    )
    inception = InceptionV3(
        weights="imagenet",
        include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        pooling="avg",
    )
    return (
        np,
        resnet,
        inception,
        resnet_preprocess,
        inception_preprocess,
    )


def extract_features(image_path: str):
    """Return the 4096-value ResNet50 + InceptionV3 hybrid feature vector."""

    rgb = preprocess_image(image_path)
    np, resnet, inception, resnet_preprocess, inception_preprocess = _get_feature_models()

    rgb = rgb.astype(np.float32)
    resnet_batch = resnet_preprocess(rgb.copy())[None, ...]
    inception_batch = inception_preprocess(rgb.copy())[None, ...]

    resnet_features = resnet.predict(resnet_batch, verbose=0)[0]
    inception_features = inception.predict(inception_batch, verbose=0)[0]

    return np.concatenate([resnet_features, inception_features])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def predict(features, bundle, *, t_severe: float | None = None, t_referable: float | None = None) -> dict:
    """Turn one feature vector into detector state.

    Args:
        features: 4096-value hybrid vector (any array-like).
        bundle: joblib bundle dict (or object with the same attributes).
        t_severe: override the severe threshold.
        t_referable: override the referable threshold.
    """

    model = _lookup(bundle, _MODEL_KEYS, "classifier")
    scaler = None
    try:
        scaler = _lookup(bundle, _SCALER_KEYS, "scaler")
    except KeyError:
        pass

    row = [features]
    if scaler is not None:
        row = scaler.transform(row)

    if not hasattr(model, "predict_proba"):
        raise TypeError("detector SVM must be fitted with probability=True")

    probs = [float(probability) for probability in model.predict_proba(row)[0]]
    if len(probs) != 5:
        raise ValueError(f"model returned {len(probs)} classes; expected 5")

    classes = getattr(model, "classes_", None)
    if classes is not None:
        class_order = [int(value) for value in classes]
        if sorted(class_order) != list(range(5)):
            raise ValueError(f"model classes are not 0-4: {class_order}")
        by_class = dict(zip(class_order, probs))
        probs = [by_class[grade] for grade in range(5)]

    raw_grade = max(range(len(probs)), key=probs.__getitem__)
    confidence = float(probs[raw_grade])

    t_severe = (
        float(t_severe)
        if t_severe is not None
        else _optional_threshold(bundle, _T_SEVERE_KEYS, settings.t_severe)
    )
    t_referable = (
        float(t_referable)
        if t_referable is not None
        else _optional_threshold(bundle, _T_REFERABLE_KEYS, settings.t_referable)
    )

    p_severe = float(sum(probs[SEVERE_GRADE:]))
    p_referable = float(sum(probs[REFERABLE_GRADE:]))

    urgent = p_severe >= t_severe
    referable = p_referable >= t_referable

    flags: list[str] = []
    if urgent:
        flags.append(
            f"urgent: P(Severe or worse)={p_severe:.2f} >= T_SEVERE={t_severe:.2f}"
        )
    if referable:
        flags.append(
            f"referable: P(Moderate or worse)={p_referable:.2f} "
            f">= T_REFERABLE={t_referable:.2f}"
        )

    grade = raw_grade
    if urgent:
        grade = max(grade, SEVERE_GRADE)
    elif referable:
        grade = max(grade, REFERABLE_GRADE)

    return {
        "raw_grade": raw_grade,
        "grade": grade,
        "label": GRADE_LABELS[grade],
        "confidence": confidence,
        "probs": probs,
        "urgent": urgent,
        "referable": referable,
        "flags": flags,
        "t_severe": t_severe,
        "t_referable": t_referable,
    }


def detect(image_path: str, *, bundle=None, extractor=extract_features) -> dict:
    """Run the full detector for one image."""

    if bundle is None:
        bundle = load_bundle()
    features = extractor(image_path)
    return predict(features, bundle)


def detector_node(state: DR_State) -> dict:
    """LangGraph node: fill the detector slice of the state."""

    if not state.image_path:
        raise ValueError("detector requires image_path")
    return detect(state.image_path)
