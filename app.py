"""Serve the HTML frontend and the locally trained classification model."""
import os
import json
import hashlib
import joblib
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from software_schema import ROOT, OUT, FEATURES, RANGES, LABELS, validate_batch

app = Flask(__name__, static_folder=str(ROOT / 'frontend'), static_url_path='/assets')
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
models = joblib.load(OUT / 'models.joblib')
details = json.loads((OUT / 'evaluation.json').read_text())
policy = json.loads((OUT / 'decision_policy.json').read_text())
if policy['model_sha256'] != hashlib.sha256((OUT / 'models.joblib').read_bytes()).hexdigest():
    raise RuntimeError('Decision policy does not match the saved models. Run software_reliability.py.')
model = models[policy['model']]

@app.get('/')
def index():
    return send_from_directory(ROOT / 'frontend', 'portal.html')

@app.get('/assessment')
def assessment_page():
    return send_from_directory(ROOT / 'frontend', 'index.html')

@app.get('/api/schema')
def schema():
    return jsonify(features=[{'key': f, 'label': LABELS[f], 'min': RANGES[f][0], 'max': RANGES[f][1]} for f in FEATURES],
                   model=policy['model'], threshold=policy['threshold'], synthetic=True,
                   task='Binary classification', classes=['Not Selected', 'Selected'])

@app.post('/api/predict')
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get('candidates'), list):
        return jsonify(error='Send a candidates list containing assessment objects.'), 400
    candidates = payload['candidates']
    return predict_candidates(candidates)

@app.post('/api/batch')
def batch():
    uploaded = request.files.get('file')
    if not uploaded:
        return jsonify(error='Choose a CSV assessment file.'), 400
    try:
        data = pd.read_csv(uploaded)
        return predict_candidates(data.to_dict(orient='records'))
    except (ValueError, pd.errors.ParserError, UnicodeError) as error:
        return jsonify(error=f'Could not read CSV: {error}'), 400

def predict_candidates(candidates):
    if not 1 <= len(candidates) <= 500 or not all(isinstance(row, dict) for row in candidates):
        return jsonify(error='Provide between 1 and 500 candidate objects.'), 400
    if any(any(isinstance(row.get(f), bool) for f in FEATURES) for row in candidates):
        return jsonify(error='Assessment scores must be numbers, not boolean values.'), 400
    try:
        inputs = validate_batch(pd.DataFrame(candidates))
    except (ValueError, TypeError) as error:
        return jsonify(error=str(error)), 400
    probabilities = model.predict_proba(inputs)[:, 1]
    results = []
    for i, score in enumerate(probabilities):
        contributions = []
        if policy['model'] == 'Logistic Regression':
            transformed = model.named_steps['scale'].transform(model.named_steps['impute'].transform(inputs.iloc[[i]]))[0]
            weights = transformed * model.named_steps['model'].coef_[0]
            contributions = sorted([{'feature': LABELS[f], 'value': float(v)} for f, v in zip(FEATURES, weights)], key=lambda x: abs(x['value']), reverse=True)
        results.append({'outcome': 'Selected' if score >= policy['threshold'] else 'Not Selected',
                        'model_score': float(score), 'threshold': policy['threshold'],
                        'contributions': contributions})
    return jsonify(results=results, model=policy['model'], synthetic=True)

@app.get('/health')
def health():
    return jsonify(status='ok')

@app.errorhandler(413)
def too_large(error):
    return jsonify(error='The request is too large. Résumés must be smaller than 5 MB; assessment batches support at most 500 candidates.'), 413

from recruitment import register_recruitment
register_recruitment(app, predict_candidates)

if __name__ == '__main__':
    from waitress import serve
    serve(app, host=os.environ.get('HOST', '127.0.0.1'), port=int(os.environ.get('PORT', '8502')))
