# Repository Inspection Readiness Plan

> **Scope:** Cosmetic and documentation-only preparation for external inspection. Do not change model definitions, training loops, adversary implementations, hyperparameters, generated outputs, or experiment semantics unless a later scientific-maintenance pass is explicitly authorised.

**Repository:** `aaronmkts/Radamacher-Compleity-of-Quantum-Classifiers`

**Paper:** [On the Generalization of Adversarially Trained Quantum Classifiers](https://arxiv.org/abs/2504.17690)

**Authors:** Petros Georgiou, Aaron Mark Thomas, Sharu Theresa Jose, Osvaldo Simeone

**Primary objective:** Make the repository understandable, navigable, and runnable by other researchers without altering how the code currently operates.

---

## 1. Current State

The repository already contains the core research code under `src/`:

- `src/codebase/datasets/` — synthetic classical data generation.
- `src/codebase/model/` — quantum classifier implementations for angle, amplitude, and reuploading embeddings.
- `src/codebase/adversary/` — classical and quantum adversarial attack utilities.
- `src/codebase/train/` — conventional and adversarial training loops.
- `src/experiments/` — experiment scripts used to generate losses, plots, and single-circuit quantum adversary results.

The main inspection gap is documentation. The previous `README.md` only contained the repository title, so an external reader could not reliably infer:

- which paper the code belongs to;
- how to install dependencies;
- which scripts are entry points;
- which scripts require pre-generated `.npy` artefacts;
- what the expected runtime/plotting behaviour is;
- what is safe to change versus what would alter the scientific implementation.

---

## 2. Non-Goals

These are deliberately out of scope for the current pass:

1. Refactoring experiment scripts.
2. Renaming modules, functions, or files.
3. Adding CLI arguments.
4. Changing default hyperparameters.
5. Changing random seeds.
6. Replacing `sys.path.append(...)` import handling.
7. Reformatting code with Black/Ruff if it could create large non-semantic diffs.
8. Re-generating paper figures or committing generated `.npy`/plot artefacts.
9. Modifying model, training, loss, or adversarial attack behaviour.
10. Applying Black formatting to source files during this pass. Use `black --check src` for inspection only unless a separate source-formatting branch is explicitly authorised.

External inspection readiness here means: *a reader can understand and run the existing code*, not that the code is redesigned into a packaged library.

---

## 3. Documentation Changes Completed in This Pass

### Task 1 — Replace the placeholder README

**File:** `README.md`

**Change:** Replace the two-line placeholder with a complete project README containing:

- paper title and arXiv link;
- repository purpose;
- installation instructions;
- Python version guidance;
- dependency notes;
- repository layout;
- experiment entry points;
- run commands;
- notes about plotting and generated artefacts;
- troubleshooting section;
- citation section;
- contribution/inspection guidelines.

**Verification:** Read the rendered Markdown and confirm all listed commands correspond to actual files in the repository.

### Task 2 — Add this readiness plan

**File:** `docs/INSPECTION_READINESS_PLAN.md`

**Change:** Record the exact external-inspection plan and constraints so future cleanup remains scoped and does not accidentally alter scientific behaviour.

**Verification:** Confirm only documentation files changed.

---

## 4. Recommended Cosmetic-Only Follow-Up Tasks

These are safe future tasks if further polish is desired.

### Task 3 — Add repository metadata on GitHub

**Location:** GitHub repository settings, not source code.

**Actions:**

- Set repository description to: `Research code for On the Generalization of Adversarially Trained Quantum Classifiers`.
- Add topics: `quantum-machine-learning`, `adversarial-training`, `rademacher-complexity`, `pennylane`, `pytorch`, `quantum-classifiers`.
- Add the arXiv URL as the repository website.

**Risk:** None to code execution.

### Task 4 — Add a lightweight `CITATION.cff`

**File:** `CITATION.cff`

**Purpose:** Make GitHub display a citation button.

**Risk:** Documentation/metadata only.

**Note:** Confirm final publication venue/DOI first if one exists. If not, use the arXiv identifier.

### Task 5 — Add an artefact manifest, if generated arrays are shared later

**File:** `docs/ARTEFACTS.md`

**Purpose:** Document every expected generated `.npy` file used by `plot.py`, `test.py`, and `quantum_single_circ.py`, including:

- generating script;
- expected shape;
- corresponding figure/table in the paper;
- whether the artefact is committed, downloadable, or intentionally omitted.

**Risk:** Documentation only.

### Task 6 — Add a minimal smoke-test note without changing code

**File:** `docs/SMOKE_TESTS.md`

**Purpose:** List the minimal commands used to verify syntax/import/runtime readiness.

**Potential commands:**

```bash
python -m compileall -q src
PYTHONPATH="$PWD/src" python - <<'PY'
from codebase.datasets import gen_data
from codebase.model import get_classifier
from codebase.adversary import get_attack
print(gen_data(4, 2)[0].shape)
print(get_classifier('angle'))
print(get_attack('FGSM'))
PY
```

**Risk:** Documentation only.

### Task 7 — Add Black formatting inspection

**Purpose:** Allow external inspectors to check Python formatting consistency without changing how the code runs.

**Command:**

```bash
black --check src
```

**Policy:** In an inspection-only pass, do not run `black src` or commit Black-generated source diffs unless a separate source-formatting branch is explicitly approved. Black formatting should be treated as a cosmetic follow-up because it may create large non-semantic changes.

**Risk:** `black --check src` is read-only. `black src` modifies Python files and is therefore outside this documentation-only pass.

---

## 5. Verification Checklist for External Release

Before directing collaborators/reviewers to the repository, verify:

- [ ] `README.md` renders correctly on GitHub.
- [ ] arXiv link resolves to the correct paper.
- [ ] listed files exist in the repository.
- [ ] dependency installation works in a fresh Python 3.10/3.11 environment.
- [ ] `python -m compileall -q src` succeeds.
- [ ] `black --check src` has been run and any formatting findings are recorded without applying source changes.
- [ ] at least one short experiment script starts successfully after dependency installation.
- [ ] README clearly warns that full experiments may be computationally expensive.
- [ ] README clearly identifies scripts requiring generated `.npy` artefacts.
- [ ] no source-code behaviour was changed in this documentation pass.

---

## 6. Suggested Future Scientific-Maintenance Pass

Only after external inspection readiness is complete, and only with explicit approval, consider a separate branch for scientific maintenance:

1. Replace hard-coded hyperparameters with optional CLI arguments while preserving defaults.
2. Move generated output into an `outputs/` directory.
3. Add deterministic smoke tests for small sample sizes.
4. Add figure reproduction scripts with explicit artefact dependencies.
5. Replace local `sys.path` manipulation with package installation or module execution.
6. Pin dependency versions once a known-good environment is confirmed.

This would improve maintainability, but it is intentionally not part of the present cosmetic/documentation pass.
