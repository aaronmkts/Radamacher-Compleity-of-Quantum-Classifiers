# Generated Artefacts

The repository contains research scripts that read and write NumPy `.npy` artefacts in the current working directory. These artefacts are outputs/intermediate data products, not source code.

## Policy

- Keep generated artefacts separate from source code unless preparing an explicit artefact release.
- Record the script and parameter settings used to generate each artefact.
- Do not assume a plotting script can run before the corresponding arrays have been generated or copied into the working directory.

## Artefacts Referenced in the Codebase

### Synthetic datasets

- `x_r=30_d=<d>_m=20.npy`
  - Meaning: generated input arrays for repeated runs.
  - Referenced by: `src/experiments/test.py`, `src/experiments/quantum_single_circ.py`, `src/experiments/plot.py`.

- `y_r=30_d=<d>_m=20.npy`
  - Meaning: generated label arrays for repeated runs.
  - Referenced by: `src/experiments/test.py`, `src/experiments/quantum_single_circ.py`.

### Loss arrays

- `angle_losses_dim_r=30_m=20_e=40.npy`
  - Meaning: loss records for angle-embedding experiments.
  - Referenced by: `src/experiments/plot.py`.

- `ampl_losses_dim_r=30_m=20_e=40.npy`
  - Meaning: loss records for amplitude-embedding experiments.
  - Referenced by: `src/experiments/plot.py`.

- `l_2_angle_losses_dim_r=30_m=20_e=40.npy`
  - Meaning: L2-adversarial angle-embedding loss records.
  - Written by: `src/experiments/test.py`.

- `l_2ampl_losses_dim_r=30_m=20_e=40.npy`
  - Meaning: L2-adversarial amplitude-embedding loss records.
  - Written by: `src/experiments/test.py`.

### Quantum adversary/theory arrays

- `noiseless_losses_epsilon_d=6_const_classifier2.npy`
- `noisy_losses_epsilon_d=6_const_classifier2.npy`
- `noiseless_adv_th.npy`
- `noiseless_th.npy`
- `noisy_adv_th.npy`
- `noisy_adv_th3.npy`
- `noisy_th.npy`
- `noisy_losses_BIM.npy`
- `noiseless_losses_BIM.npy`

These are referenced or written by `src/experiments/quantum_single_circ.py` and/or loaded by `src/experiments/plot.py`.

### Training/test loss arrays

- `train_loss.npy`
- `test_loss.npy`
- `adv_train_loss.npy`
- `adv_test_loss.npy`

These are written by `src/experiments/test.py` and can be used for plotting/analysis.

## Common Failure Mode

If a script fails with:

```text
FileNotFoundError: *.npy
```

then the relevant generated artefact is missing from the current working directory. Generate it with the corresponding experiment script or place the expected file in the repository root before running the plotting/analysis path.

## Suggested Release Practice

For a future formal artefact release, add a versioned archive or external storage link containing:

- generated `.npy` arrays;
- figure outputs;
- script/parameter provenance;
- Python/package versions;
- expected shapes for each array.
