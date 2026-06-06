# Rademacher Complexity of Quantum Classifiers

Research code accompanying the paper:

**On the Generalization of Adversarially Trained Quantum Classifiers**  
Petros Georgiou, Aaron Mark Thomas, Sharu Theresa Jose, Osvaldo Simeone  
arXiv: [2504.17690](https://arxiv.org/abs/2504.17690)

> This repository contains Python/PennyLane/PyTorch code for numerical experiments on the generalization behaviour of conventional and adversarially trained quantum classifiers.

---

## Overview

Quantum classifiers can be vulnerable to adversarial perturbations of either the classical input data or the quantum states encoding those inputs. The associated paper develops generalization bounds for adversarially trained quantum classifiers and validates the theoretical findings using numerical experiments.

This codebase includes:

- synthetic classical data generation;
- angle-embedding and amplitude-embedding quantum classifiers;
- conventional and adversarial training loops;
- classical-input adversarial attacks;
- quantum-state adversarial attack utilities;
- experiment scripts for generating losses and plotting results.

The repository is primarily a research-code release. It is intended to make the experiments inspectable and reproducible, rather than to provide a polished Python package or command-line application.

---

## Repository Layout

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── docs/
│   └── INSPECTION_READINESS_PLAN.md
└── src/
    ├── codebase/
    │   ├── adversary/      # Adversarial attacks for classical and quantum settings
    │   ├── datasets/       # Synthetic Gaussian data generation
    │   ├── model/          # Quantum classifier definitions
    │   └── train/          # Conventional and adversarial training loops
    └── experiments/        # Experiment and plotting scripts
```

Key modules:

- `src/codebase/datasets/classical_data.py` — generates synthetic binary-class Gaussian data.
- `src/codebase/model/angle.py` — angle-embedding classifier.
- `src/codebase/model/amplitude.py` — amplitude-embedding classifier.
- `src/codebase/model/reuploading.py` — reuploading classifier.
- `src/codebase/adversary/` — FGSM, PGD, L1/L2, and quantum adversary routines.
- `src/codebase/train/base_train.py` — standard training loop.
- `src/codebase/train/adversary_train.py` — adversarial training loop.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/aaronmkts/Radamacher-Compleity-of-Quantum-Classifiers.git
cd Radamacher-Compleity-of-Quantum-Classifiers
```

The repository name currently uses the historical spelling `Radamacher-Compleity-of-Quantum-Classifiers`. The mathematical term is *Rademacher complexity*.

### 2. Create a Python environment

Python 3.10 is the safest target because the included `environment.yml` was created with Python 3.10. The code also syntax-checks under Python 3.11.

Using `venv`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `python3.10` is not installed, Python 3.11 may also work:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Using Conda:

```bash
conda create -n quclassifier python=3.10 pip
conda activate quclassifier
pip install -r requirements.txt
```

Notes:

- `requirements.txt` contains the main runtime dependencies: `scikit-learn`, `pennylane`, `torch`, `matplotlib`, `datasets`, and `ipdb`.
- Some adversary/quantum experiment paths additionally import `scipy` and `numdifftools`. Install them before running the full experiment suite or the adversary import smoke test:

```bash
pip install scipy numdifftools
```

- The checked-in `environment.yml` records a local development environment and does not install the full Python dependency set by itself. For a fresh setup, prefer the `pip install -r requirements.txt` route above, then install the extra packages above if you need the quantum adversary paths.
- For optional formatting/lint-style inspection, install Black:

```bash
pip install black
```

---

## Basic Usage

Run commands from the repository root. The experiment scripts add `src/` to `sys.path` themselves, but setting `PYTHONPATH` explicitly is useful for interactive work:

```bash
export PYTHONPATH="$PWD/src"
```

### Quick syntax check

This verifies that all Python files parse correctly without running the experiments:

```bash
python -m compileall -q src
```

Expected result: no output and exit code `0`.

### Import smoke test

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

Expected output includes:

```text
x shape: (4, 2)
y shape: (4,)
angle classifier: <class 'codebase.model.angle.AngleEmbeddingClassifier'>
FGSM attack: <function FGSM ...>
```

Exact function/class formatting may differ across Python versions.

### Formatting/lint-style inspection with Black

This repository currently preserves the original research-code formatting. To inspect formatting without changing source files, run Black in check mode:

```bash
black --check src
```

If Black reports files that would be reformatted, treat that as cosmetic information. Do not run `black src` as part of an inspection-only pass unless a source-formatting branch has been explicitly approved, because it will modify Python files even though it should not change execution semantics.

---

## Experiment Scripts

The scripts in `src/experiments/` are the main entry points. They are configured through constants inside each script rather than command-line arguments.

### Standard generalization experiment

```bash
python src/experiments/non_adv_train_gen.py
```

Purpose:

- generates synthetic data;
- trains conventional quantum classifiers;
- compares train/test loss differences across embedding types and dimensions;
- plots the resulting loss-difference curves.

Main settings are defined near the top of the file:

- `dimensions_list = [2, 4, 8, 16]`
- `embeddings_list = ['angle', 'amplitude']`
- `epochs = 5`
- `train_samples = 500`
- `test_samples = 2000`
- `batch_size = 10`

### Adversarial generalization experiment

```bash
python src/experiments/adv_train_gen.py
```

Purpose:

- generates synthetic data;
- trains quantum classifiers using adversarial examples;
- evaluates adversarial train/test loss gaps;
- plots the resulting loss-difference curves.

Main settings include:

- `dimensions_list = [2, 4, 8, 16]`
- `embeddings_list = ['angle', 'amplitude']`
- `attack = 'l_1'`
- `epsilon = 0.05`
- `train_samples = 500`
- `test_samples = 2000`

### Self-contained standard classifier script

```bash
python src/experiments/non_adv_train_sc.py
```

Purpose:

- defines a classifier directly inside the script;
- generates synthetic data;
- trains a standard quantum classifier;
- plots loss and test accuracy.

This script is useful for inspecting the training loop without relying as heavily on the package-style modules under `src/codebase/`.

### Self-contained adversarial classifier script

```bash
python src/experiments/adv_train_sc.py
```

Purpose:

- defines a classifier and adversarial training routine directly inside the script;
- compares loss gaps for angle and amplitude embeddings;
- uses an adversary selected by the `attacker` variable.

Default setting:

```python
attacker = "l_2"
epsilon = 0.05
```

### Adversarial dataset batch smoke script

```bash
python src/experiments/batch_test.py
```

Purpose:

- creates a small angle-embedding classifier;
- generates synthetic data;
- applies an adversarial attack to a batch;
- prints the number of generated adversarial inputs.

This is a comparatively lightweight script for checking that the adversarial dataset path starts up.

### Quantum-state adversary experiments

```bash
python src/experiments/quantum_single_circ.py
```

Purpose:

- experiments with quantum-state perturbations and density-matrix based adversarial attacks;
- contains multiple helper routines for noisy/noiseless comparisons and theory-bound artefacts.

Important notes:

- This script imports `scipy`, so install it with `pip install scipy` if needed.
- Some code paths expect generated `.npy` files such as `x_r=30_d=6_m=20.npy` and `y_r=30_d=6_m=20.npy` to exist in the working directory.
- Full runs may be substantially more expensive than the smaller classical-input experiments.

### Plotting script

```bash
python src/experiments/plot.py
```

Purpose:

- loads previously generated `.npy` arrays;
- produces paper-style plots using Matplotlib and LaTeX-style formatting.

Important notes:

- `plot.py` expects generated arrays such as:
  - `ampl_losses_dim_r=30_m=20_e=40.npy`
  - `angle_losses_dim_r=30_m=20_e=40.npy`
  - `x_r=30_d=<d>_m=20.npy`
- These artefacts are not currently committed to the repository.
- Matplotlib is configured with `text.usetex = True`, so a local LaTeX installation may be required for final figure rendering.

---

## Generated Artefacts

Several experiment scripts read or write `.npy` files in the current working directory. These files are treated as generated research artefacts rather than source files.

Examples referenced in the code include:

```text
x_r=30_d=<d>_m=20.npy
y_r=30_d=<d>_m=20.npy
angle_losses_dim_r=30_m=20_e=40.npy
ampl_losses_dim_r=30_m=20_e=40.npy
noiseless_losses_epsilon_d=6_const_classifier2.npy
noisy_losses_epsilon_d=6_const_classifier2.npy
noiseless_adv_th.npy
noiseless_th.npy
noisy_adv_th.npy
noisy_adv_th3.npy
noisy_th.npy
train_loss.npy
test_loss.npy
adv_train_loss.npy
adv_test_loss.npy
```

If a script fails with `FileNotFoundError` for one of these files, generate the relevant artefact first or place the expected `.npy` file in the repository root before running the plotting/analysis script.

---

## Reproducibility Notes

Most scripts set seeds with:

```python
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
```

However, exact numerical reproducibility can still depend on:

- Python version;
- PyTorch version;
- PennyLane version;
- CPU/GPU backend;
- BLAS/OpenMP behaviour;
- whether a script uses generated `.npy` artefacts from a previous run.

For inspection, start with the syntax/import smoke tests before launching full experiments.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'codebase'`

Run from the repository root and set:

```bash
export PYTHONPATH="$PWD/src"
```

Alternatively, run the scripts exactly as shown above from the repository root.

### `ModuleNotFoundError: No module named 'pennylane'` or `No module named 'torch'`

Install dependencies:

```bash
pip install -r requirements.txt
```

### `ModuleNotFoundError: No module named 'scipy'` or `No module named 'numdifftools'`

Some quantum-state and adversary paths use SciPy and Numdifftools. Install them manually:

```bash
pip install scipy numdifftools
```

### `FileNotFoundError: *.npy`

The plotting and some quantum-state experiment paths expect generated `.npy` artefacts. Those files are not source code and may need to be generated separately before plotting.

### Matplotlib/LaTeX rendering errors

`src/experiments/plot.py` enables LaTeX text rendering. If LaTeX is unavailable, either install a TeX distribution or inspect the numerical arrays without using the final plotting script.

### Long runtimes

Quantum circuit simulation and repeated adversarial training can be slow. Start with small smoke tests before full experiment runs.

---

## Development and Inspection Guidelines

For external inspection of the current repository:

1. Treat the code as research code corresponding to the arXiv paper.
2. Avoid changing training loops, attacks, model definitions, loss functions, or hyperparameters unless reproducing a specific experiment requires it.
3. Prefer documentation-only changes for public-readiness work.
4. Keep generated artefacts separate from source code unless an explicit artefact release is prepared.
5. Record any environment/version changes when reporting reproduced results.

A cosmetic/documentation-only readiness plan is available at:

```text
docs/INSPECTION_READINESS_PLAN.md
```

---

## Citation

If you use this repository, please cite the paper:

```bibtex
@misc{georgiou2025generalizationadversariallytrainedquantum,
  title         = {On the Generalization of Adversarially Trained Quantum Classifiers},
  author        = {Petros Georgiou and Aaron Mark Thomas and Sharu Theresa Jose and Osvaldo Simeone},
  year          = {2025},
  eprint        = {2504.17690},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  url           = {https://arxiv.org/abs/2504.17690}
}
```

---

## License

This repository is released under the MIT License. See [`LICENSE`](LICENSE).
