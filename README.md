## MAI IDSS W3 — Group 3: Wildfire Risk Project

This repository will implement the project described in our proposal document. For full context, see the project proposal in `docs/IDSS_W3-Project_Proposal-Group3.pdf`.

### What this repo is for

- Central place for the code, data setup, and documentation to reproduce the analyses and models outlined in the proposal.
- The raw dataset is sourced from Kaggle and should be placed in the `data/` directory as a CSV file (details below).

### Repository structure

- `src/backend/` — APIs, data processing pipelines, training jobs.
- `src/frontend/` — UI components, dashboards, visualizations.
- `models/` — Saved model artifacts and experiment outputs.
- `data/` — Local data folder (excluded from version control for large/raw files). Place the raw Kaggle CSV here.
- `docs/` — Project documents (proposal, notes, figures). See `IDSS_W3-Project_Proposal-Group3.pdf`.

## Data setup (Kaggle)

You’ll need a Kaggle account to download the dataset. There are two supported ways: manual download from the website or using the Kaggle CLI. On Windows, the commands below assume PowerShell.

Before you start, decide on the final local filename for the raw data CSV. We expect it at:

- `data/Wildfire_Dataset.csv`

If the downloaded file name differs, just rename it to match the expected path above, or update any downstream scripts accordingly once they exist.

### Option A — Manual download from kaggle.com

1) Sign in to Kaggle and navigate to the dataset page: https://www.kaggle.com/datasets/firecastrl/us-wildfire-dataset
2) Click “Download” to get the CSV (or a ZIP containing the CSV).
3) If you get a ZIP, extract it.
4) Move the CSV into this repository’s `data/` folder and, if needed, rename it to `Wildfire_Dataset.csv`.

### Option B — Kaggle CLI (PowerShell)

1) Install Python and pip if you don’t have them.
2) Install the Kaggle CLI:

```powershell
pip install kaggle
```

3) Create and place your Kaggle API token:

- In your Kaggle account, go to Account settings → “Create New API Token”. This downloads `kaggle.json`.
- Move it to the expected location for the CLI on Windows:

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.kaggle" | Out-Null
Move-Item -Force "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"
```

4) Download the dataset to the `data/` folder using the specific dataset slug `firecastrl/us-wildfire-dataset`:

```powershell
# Download the dataset archive into data/
kaggle datasets download -d firecastrl/us-wildfire-dataset -p data

# If a ZIP is downloaded, unzip then rename if necessary
Expand-Archive -Path "data\*.zip" -DestinationPath data -Force
Remove-Item "data\*.zip"

# If the extracted CSV name differs, rename it to the expected filename
# Replace <downloaded-name>.csv with the actual extracted filename
Rename-Item -Path "data\<downloaded-name>.csv" -NewName "Wildfire_Dataset.csv"
```

Notes:

- The Kaggle CLI may warn about file permissions on Windows; this warning is safe to ignore.
- Keep your `kaggle.json` private (don’t commit it). It contains your Kaggle API credentials.

## Next steps

- Once the dataset is placed in `data/Wildfire_Dataset.csv`, upcoming scripts and notebooks in `src/` will load it from there for preprocessing, EDA, and modeling as per the proposal.
- We’ll add run instructions, environment setup, and evaluation details as components land in this repository.

## Referencing the proposal

For scope, objectives, features, and metrics, see the proposal PDF:

- `docs/IDSS_W3-Project_Proposal-Group3.pdf`

This README will evolve as we implement the plan described in that document.

