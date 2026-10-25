# CandidateLab

Interview Candidate Selection Predictor for a software-engineering assessment, with a Jupyter notebook and an HTML/CSS/JavaScript interface served by Flask. The model classifies a candidate as **Selected** or **Not Selected**. HR can manage candidate records and assessments; project managers can match skills to project requirements and maintain shortlists.

**Academic demonstration:** the dataset contains fictional, synthetic candidates. Its results do not establish real-world hiring accuracy or fairness. Use fictional details and sample resumes when demonstrating the application.

## Start the application

Python 3.14 was used for the saved project model. Create a virtual environment to keep the pinned dependencies separate from other assignments.

```powershell
git clone https://github.com/WhitefistEmperor/CandidateLab.git
cd CandidateLab
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:8502/** in your browser. On macOS/Linux, activate the environment with `source .venv/bin/activate` instead.

| Local demonstration role | Username | Password |
|---|---|---|
| HR | `hr` | `hr123` |
| Project manager | `manager` | `manager123` |

These are local demonstration defaults, not the private cloud credentials. New candidate records, resumes and logs stay in the ignored `instance/` folder. A fresh clone starts without anyone else's local candidate records. Configure `SECRET_KEY`, `HR_PASSWORD` and `MANAGER_PASSWORD` before hosting outside a local demonstration.

## Open the Jupyter notebook

```powershell
python -m pip install -r requirements-notebook.txt
python -m jupyterlab Candidate_Selection_Classification.ipynb
```

The notebook already includes executed outputs and graphs. To run it, open it from this complete project folder and run all cells from top to bottom. Downloading only the notebook is enough to read its saved outputs, but running it requires the dataset, helper modules and saved model included in this repository.

The notebook covers:

- Manual cleaning with Pandas, including numeric checks, duplicate removal and review records.
- A stratified 80/20 training/test split and three classifiers: Logistic Regression, Decision Tree and Random Forest.
- Five-fold cross-validation for model comparison, followed by held-out test evaluation.
- Accuracy, precision, recall, F1, confusion matrices and a majority-class baseline.
- Candidate prediction, overfitting checks and verification of the shared saved model.
- Assignment-style bar charts and coloured confusion matrices in Section 10.

There are 48 notebook cells, including 21 Python cells. The raw cleaning exercise contains 9,200 rows: 9,000 synthetic records plus 200 duplicate copies. Cleaning leaves 8,980 modelling records, split into 7,184 training and 1,796 test records.

## Model and decision policy

`results/software/models.joblib` contains the saved classifiers. The active model is Logistic Regression, selected using training cross-validation macro F1. The application checks the model-file hash against `results/software/decision_policy.json`.

The notebook's ordinary test predictions and Section 10 test charts use the default **0.50** threshold. The application's separate saved policy uses **0.68**, selected from training-validation predictions to favour precision. Do not report the 0.50 test metrics as performance of the 0.68 policy.

The notebook verifies that its training run reproduces the saved models. It saves metric tables without replacing the active model or policy. Jupyter is not executed for each website request; Flask loads the saved model and performs inference.

Only load joblib/pickle model files from sources you trust. This repository includes the project's own model artifact.

## Presentation guides

| Guide | Assigned AIML explanation |
|---|---|
| [Your guide](AIML_Guide_You.docx) | Training, cross-validation, evaluation and overfitting; cells 17-26 and 29-30 |
| [R guide](AIML_Guide_R.docx) | Predictions, verification, graphs and conclusion; cells 27-28 and 31-48 |
| [S guide](AIML_Guide_S.docx) | Introduction, manual cleaning and train/test split; cells 1-16 |

Cell numbers count both text and code blocks from 1; execution counts shown as `In [ ]` can change.

## Tests and rebuilding

```powershell
python -m unittest test_api.py test_recruitment.py test_software_project.py -v
```

The recruitment tests use temporary storage. The API tests use the default local demo accounts. Avoid running these tests against an environment configured with cloud credentials or real candidate data.

To intentionally regenerate the synthetic data, retrain the models and recompute the policy:

```powershell
python run_project.py
```

This rebuild overwrites generated data and model outputs. It is not required to launch the application or read the notebook. If you change training or model selection, revalidate the decision policy before using the application.

## Docker and optional cloud storage

```powershell
docker compose up --build -d
```

Open http://127.0.0.1:8502/. Compose keeps the candidate database in a named volume and binds the website to localhost. `docker compose down` stops the local container; adding `-v` also deletes that volume's records.

The app supports private Amazon S3 resume storage through the optional dependencies in `requirements-cloud.txt`. `cloud.env.example` contains placeholders only. The existing private AWS deployment's credentials, account-specific templates, live addresses, screenshots and status records are excluded from this public repository. Cloning the repository does not grant access to that AWS account or deployment.

## Main files

| File or folder | Purpose |
|---|---|
| `Candidate_Selection_Classification.ipynb` | Main teaching notebook with executed visualisations |
| `app.py`, `recruitment.py` | Prediction API, authentication and recruitment workflow |
| `frontend/` | HTML, CSS and JavaScript interface |
| `software_schema.py` | Features, labels, score ranges and validation |
| `generate_software_data.py`, `clean_software_data.py` | Reproducible synthetic data and cleaning |
| `software_ml.py`, `software_reliability.py` | Training, evaluation and threshold development |
| `notebook_support.py` | Saved-model and prediction consistency checks |
| `data/software/` | Synthetic dataset and cleaning reports |
| `results/software/` | Saved model, policy and evaluation artifacts |
| `results/notebook/` | Notebook metric tables and plotted results |

See [JUPYTER_START.md](JUPYTER_START.md), [SOFTWARE_PROJECT_GUIDE.md](SOFTWARE_PROJECT_GUIDE.md) and [DATA_SOURCE_REVIEW.md](DATA_SOURCE_REVIEW.md) for more explanation. Historical Streamlit prototypes and unrelated third-party datasets are not part of this public project.
