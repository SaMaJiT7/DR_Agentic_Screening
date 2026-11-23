# Test suite

Command:

```bash
for f in tests/test_*.py; do PYTHONPATH=. /Users/samajit/Projects/Diabetic_agentic/.venv-ml/bin/python "$f" || exit 1; done
```

Output:

```text
test_care_kb.py                  PASS  (care KB tests passed)
test_caregiver.py                PASS  (care-giver tests passed)
test_cost.py                     PASS  (cost tests passed)
test_cost_kb.py                  PASS  (cost KB tests passed)
test_detector.py                 PASS  (detector tests passed)
test_explainer.py                PASS  (explainer tests passed)
test_graph.py                    PASS  (graph tests passed)
test_hospital_search.py          PASS  (hospital_search tests passed)
test_report.py                   PASS  (report tests passed)
test_router.py                   PASS  (router tests passed)
test_screen_only.py              PASS  (screen_only tests passed)
test_stage_kb.py                 PASS  (stage KB tests passed)
test_state.py                    PASS  (state smoke tests passed)

13/13 test files passed in 1.4s
```
