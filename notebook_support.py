"""Read-only checks of the shared project model for the teaching notebook."""
from pathlib import Path
import json
import hashlib
import joblib
import numpy as np
from software_schema import ROOT as PROJECT, OUT

def save_project_models(models, selected_name, X, metrics, training_table):
    saved_models = joblib.load(OUT / 'models.joblib')
    policy = json.loads((OUT / 'decision_policy.json').read_text())
    assert selected_name == policy['model'], 'The selected algorithm changed. Revalidate the policy before deployment.'
    for name, fitted in models.items():
        np.testing.assert_allclose(fitted.predict_proba(X), saved_models[name].predict_proba(X), rtol=0, atol=1e-12)
    # Verify the classifier without replacing the active model or decision policy.
    assert policy['model_sha256'] == hashlib.sha256((OUT / 'models.joblib').read_bytes()).hexdigest()
    notebook_output = PROJECT / 'results' / 'notebook'
    notebook_output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(notebook_output / 'test_metrics.csv', index=False)
    training_table.to_csv(notebook_output / 'training_metrics.csv', index=False)
    print('Shared classifier file:', OUT / 'models.joblib')
    print('Active classifier:', selected_name)
    print('App and notebook threshold:', policy['threshold'])
    print('Verified only: the project model and policy were not overwritten.')

def check_project_prediction(example):
    import pandas as pd
    from app import app, predict_candidates
    policy = json.loads((OUT / "decision_policy.json").read_text())
    model = joblib.load(OUT / "models.joblib")[policy["model"]]
    score = float(model.predict_proba(example)[0, 1])
    predicted = "Selected" if score >= policy["threshold"] else "Not Selected"
    with app.test_request_context():
        result = predict_candidates(example.to_dict(orient="records")).get_json()["results"][0]
    assert result["outcome"] == predicted
    assert abs(result["model_score"] - score) < 1e-12
    return pd.DataFrame([{"Predicted selection": predicted, "Model score": round(score, 3), "Threshold": policy["threshold"], "Matches app": True}])
