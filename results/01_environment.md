# Environment

Command:

```bash
/Users/samajit/Projects/Diabetic_agentic/.venv-ml/bin/python - <<'PY'
import platform, sys
import numpy, cv2, sklearn, tensorflow as tf, keras, langgraph
print('python      ', sys.version.split()[0])
print('platform    ', platform.platform())
print('numpy       ', numpy.__version__)
print('opencv      ', cv2.__version__)
print('scikit-learn', sklearn.__version__)
print('tensorflow  ', tf.__version__)
print('keras       ', keras.__version__)
PY
```

Output:

```text
python       3.12.16
platform     macOS-27.0.1-arm64-arm-64bit
numpy        2.5.4
opencv       5.0.0
scikit-learn 1.6.1
tensorflow   2.21.0
keras        3.15.1
```
