# AI-READI Processing Guide

A guide to generate the processed AI-READI files used by this repository's real-data notebooks.

## External Input

The data is available from the AI-READI study upon request:

- dataset documentation: [docs.aireadi.org](https://docs.aireadi.org/)
- project website: [aireadi.org](https://aireadi.org/)
- historical v2.0.0 documentation: [AI-READI docs v2.0.0](https://docs.aireadi.org/docs/2/about)

This repository expects the raw AI-READI export in a sibling directory:

```text
../dataset/
```

Relative to the repository root, the processing script looks for:

- `../dataset/participants.tsv`
- `../dataset/clinical_data/measurement.csv`
- `../dataset/wearable_blood_glucose/manifest.tsv`
- the CGM JSON files referenced by `manifest.tsv`

## Processing Step

Run:

```powershell
python tools/processing_aireadi.py
```

Optional overrides:

```powershell
python tools/processing_aireadi.py --raw-data-dir ..\dataset --output-dir .\data
```

The default behavior is:

- read the raw AI-READI export from sibling `../dataset/`
- write processed outputs into this repository's `data/` directory

## Output Files

The script writes the pinned files consumed by the notebooks:

- `data/ai-ready.csv`
- `data/2025-10-06_metadata.csv`

The pinned metadata filename is intentional. Downstream notebooks and loaders use this fixed filename for reproducibility.

## Notes

- `tools/processing_aireadi.py` is both the raw-data processing script and the home of the AI-READI-specific cohort loader used by notebooks.
- The processed cohort loader applies the repository's fixed study-group restriction and HbA1c filtering when constructing the subject-level analytic cohort.
- After generating the two files above, the AI-READI notebooks can be run directly from the repository root.
