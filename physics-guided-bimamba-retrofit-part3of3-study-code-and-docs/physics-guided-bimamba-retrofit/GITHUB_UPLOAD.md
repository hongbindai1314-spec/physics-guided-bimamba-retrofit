# Repository upload instructions

Recommended repository name: `physics-guided-bimamba-retrofit`

## Browser upload

1. Create a **Public** repository.
2. Do **not** add a second README, licence, or `.gitignore`; those files are already included.
3. Unzip the package locally.
4. Upload the **contents** of the `physics-guided-bimamba-retrofit/` folder to the repository root.
5. Commit with a message such as `Reproducibility package v1.1.0`.

## Git command line

```bash
git init
git add .
git commit -m "Reproducibility package v1.1.0"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/physics-guided-bimamba-retrofit.git
git push -u origin main
```

## Before creating the release tag

Run:

```bash
pytest -q
python scripts/run_validation_analysis.py
python scripts/run_simulation.py --mode verify
```

Then verify `README.md`, `REPRODUCIBILITY.md`, and `docs/DATA_DICTIONARY.md`. Fill in the GitHub owner and the Zenodo DOI in `CITATION.cff` after the archival release.
