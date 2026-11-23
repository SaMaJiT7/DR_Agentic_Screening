# Model bundle validation

Command:

```bash
/Users/samajit/Projects/Diabetic_agentic/.venv-ml/bin/python scripts/inspect_bundle.py dr_svm_bundle.joblib
```

Output:

```text
bundle:     /Users/samajit/Projects/Diabetic_agentic/dr_svm_bundle.joblib
python:     3.12.16
numpy:      2.5.4
scikit:     1.6.1
tensorflow: 2.21.0

keys: ['classes', 'model', 't_referable', 't_severe']
classifier key: model
classifier type: Pipeline
classes_: [0, 1, 2, 3, 4]
n_features_in_: 4096
t_severe: 0.08399999999999996   (settings fallback: 0.3)
t_referable: 0.368   (settings fallback: 0.5)
bundle classes: ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']

labels: 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative

bundle looks compatible with the detector.
```
