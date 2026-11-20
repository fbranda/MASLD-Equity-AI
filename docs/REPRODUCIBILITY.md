# Reproducibility and provenance

## Input sources

All 14 inputs are NHANES public-use SAS transport files (.xpt). The full list and source URLs are in [DATA_FILES.txt](DATA_FILES.txt). These datasets were supplied to the research workflow by the researcher and were not altered to achieve the reported estimates.

The SHA-256 fingerprints in [XPT_SHA256_AS_UPLOADED.txt](XPT_SHA256_AS_UPLOADED.txt) document the *uploaded versions* only; they are not authoritative checksums independently issued by CDC.

## Workflow

```bash
python -m pip install -r requirements.txt
sh run_all.sh
```

Equivalently, execute the stages in the following order:

```bash
python scripts/run_analysis.py --data data --output results --download --bootstrap 450
python scripts/augment_analysis.py --data data --results results --bootstrap 450
python scripts/make_figures.py --data data --results results
python scripts/build_full_paper.py --results results --out BioMedInformatics_MASLD_Reproduced_Manuscript.docx
```

The pipeline uses `SEED = 20261119`. Model selection is deterministic conditional on dependency versions, but floating-point and ML implementation changes can alter results slightly.

## Environment used for archived results

- Python 3.13.5
- NumPy 2.3.5
- pandas 2.2.3
- SciPy 1.17.0
- scikit-learn 1.8.0
- Matplotlib 3.10.8
- python-docx 1.2.0

`requirements.txt` sets compatible version ranges, not a bitwise-reproducible lockfile. Fix exact dependencies and inspect actual package versions for strict numeric replication.

## Outputs and privacy

`results/results.json` contains aggregate model metrics, weights, subgroup audits, and uncertainty estimates. Plots generated from the analysis are archived in `figures/` for quick inspection. The local runtime also generates `results/validation_predictions.csv`, which contains individual-level predicted values keyed to public-use participant identifiers; this file is excluded by `.gitignore` and must not be committed to the public repo.

The manuscript generation script requires aggregate JSON results and expected figure files under `results/`. Initial `run_all.sh` execution populates them. The generated Word manuscript remains local and is ignored by Git, pending review by all contributing authors.

## Reproduction checklist

1. Verify all 14 `.xpt` input file names and their sizes against CDC download pages.
2. Confirm NHANES variables and release-specific weights against official codebooks.
3. Run `sh run_all.sh` without changing model specifications.
4. Check cohort counts: 5,604 / 3,736; endpoint events 229 / 206.
5. Inspect aggregate outputs against the `results/results.json` archive.
6. Inspect ROC, calibration, and equity plots and verify bootstrap settings.
7. Confirm any updated manuscript claims against freshly generated outputs.
8. Repeat numerical calculations independently before publication.

## Scope of verification

The GitHub Actions workflow compiles scripts and checks consistency of the archived JSON; it **cannot run the full data workflow without internet / CDC source files**. CI passing is not proof of external validity or clinical effectiveness.
