# Single-image inference

Command:

```bash
/Users/samajit/Projects/Diabetic_agentic/.venv-ml/bin/python scripts/check_inference.py test_images/0005cfc8afb6.png
```

Output:

```text
image: test_images/0005cfc8afb6.png
wall clock: 4.3s (includes model warm-up)

image:  test_images/0005cfc8afb6.png
bundle: dr_svm_bundle.joblib

finished in 4.0s
raw grade:   1 (Mild)  confidence=0.612
final grade: 3 (Severe)  <- escalated by safety thresholds

probabilities:
  0 No DR          0.043
  1 Mild           0.612
  2 Moderate       0.250
  3 Severe         0.014
  4 Proliferative  0.081

urgent:    True  P(Severe+)=0.095  T_SEVERE=0.084
referable: False  P(Moderate+)=0.345  T_REFERABLE=0.368

flags:
  - urgent: P(Severe or worse)=0.10 >= T_SEVERE=0.08
```
