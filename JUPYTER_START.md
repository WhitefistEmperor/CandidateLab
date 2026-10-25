# Open and demonstrate the classifier

From the project folder:

```powershell
python -m pip install -r requirements-notebook.txt
python -m jupyterlab Candidate_Selection_Classification.ipynb
```

Select the Python 3 kernel, then Run → Run All Cells. The saved notebook already contains executed outputs, so it also opens as a completed demonstration.

The notebook loads raw data, cleans it directly with Pandas, defines X and y, reserves 20% for testing, trains Logistic Regression, Decision Tree and Random Forest with training-only cross-validation, and selects a model before evaluating the test set. Section 10 adds assignment-style bar charts for class counts and model metrics, a confusion matrix for each classifier, and a training-versus-cross-validation comparison. The plots use existing results and do not change the trained models.

The notebook verifies its classifiers against `results/software/models.joblib`, the artifact loaded by the app. It does not overwrite that artifact or the decision policy. The project prediction check uses the shared policy and compares it with the application's prediction function locally. Metric tables are saved under `results/notebook/`. Section 5 and the test plots use the default 0.50 classification threshold; the application's separate decision policy uses 0.68. Run the app with `python app.py` to open the interface.

The labels in this software dataset remain synthetic. Training and testing are real computations on those records, but their accuracy does not establish real-world hiring validity. No matching independently recorded software-interview outcome dataset was verified during the source audit. The notebook cites the research datasets checked and states this limitation.
