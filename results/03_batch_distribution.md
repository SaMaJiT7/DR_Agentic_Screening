# Detector distribution over 40 images

Command:

```bash
/Users/samajit/Projects/Diabetic_agentic/.venv-ml/bin/python scripts/check_folder.py test_images --limit 40 --csv results/batch_predictions.csv
```

Output:

```text
folder: test_images
images: 40

scored 40 images in 7.8s (0.19s each)

raw grades (SVM argmax):
  0 No DR              7   17.5%
  1 Mild               3    7.5%
  2 Moderate          28   70.0%
  3 Severe             1    2.5%
  4 Proliferative      1    2.5%

final grades (after t_severe / t_referable):
  0 No DR              6   15.0%
  1 Mild               2    5.0%
  2 Moderate           5   12.5%
  3 Severe            26   65.0%
  4 Proliferative      1    2.5%

escalated by thresholds: 26/40 (65.0%)
thresholds used: T_SEVERE=0.084  T_REFERABLE=0.368
settings fallback: T_SEVERE=0.3  T_REFERABLE=0.5 (bundle values win when present)

wrote 40 rows to /Users/samajit/Projects/Diabetic_agentic/results/batch_predictions.csv
```
