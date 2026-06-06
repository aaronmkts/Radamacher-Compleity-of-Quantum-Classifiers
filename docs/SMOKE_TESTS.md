# Smoke Tests

These commands provide lightweight verification that the repository is installable, importable, and syntactically valid before launching full experiments.

Run commands from the repository root.

## 1. Create and activate an environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Syntax check

```bash
python -m compileall -q src
```

Expected result: no output and exit code `0`.

## 3. Import smoke test

```bash
PYTHONPATH="$PWD/src" python - <<'PY'
from codebase.datasets import gen_data
from codebase.model import get_classifier
from codebase.adversary import get_attack

x, y = gen_data(4, 2)
print("x shape:", tuple(x.shape))
print("y shape:", tuple(y.shape))
print("angle classifier:", get_classifier("angle"))
print("FGSM attack:", get_attack("FGSM"))
PY
```

Expected output includes shapes `(4, 2)` and `(4,)`, plus the angle classifier class and FGSM attack function.

## 4. Lightweight adversarial batch script

```bash
PYTHONPATH="$PWD/src" python src/experiments/batch_test.py
```

Expected output:

```text
10
```

## 5. Black formatting check

```bash
black --check src
```

After the approved Black-formatting pass, expected result:

```text
All done! ✨ 🍰 ✨
```

## Notes

- These smoke tests do not reproduce all paper figures.
- Full experiments may be substantially slower because they simulate quantum circuits and adversarial training loops.
- Plotting scripts may require generated `.npy` artefacts and a local LaTeX installation.
