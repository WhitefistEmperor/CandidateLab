"""Freeze a training-only threshold; evaluate an independent synthetic batch."""
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix
from software_schema import DATA, OUT, FEATURES
from generate_software_data import generate, corrupt
from clean_software_data import clean

def wilson(successes, total):
    if total == 0:
        return [None, None]
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z*z/total
    midpoint = (p + z*z/(2*total))/denominator
    half = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total))/denominator
    return [float(midpoint-half), float(midpoint+half)]

def measure(y, scores, threshold):
    pred = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    n = len(y)
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
    # Nonparametric row-bootstrap of binary outcome counts, equivalent to sampling rows.
    bootstrap = np.random.default_rng(2718).multinomial(n, [tn/n, fp/n, fn/n, tp/n], size=1000)
    btn, bfp, bfn, btp = bootstrap.T
    samples = np.divide(2*btp, 2*btp+bfp+bfn, out=np.zeros(1000), where=(2*btp+bfp+bfn)>0)
    return {'threshold': float(threshold), 'rows': n, 'accuracy': float((tn+tp)/n),
        'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
        'accuracy_95_ci': wilson(tn+tp, n), 'precision_95_ci': wilson(tp, tp+fp),
        'recall_95_ci': wilson(tp, tp+fn), 'f1_95_ci': np.quantile(samples, [0.025, 0.975]).tolist(),
        'predicted_selected': int(tp+fp), 'true_selected': int(tp+fn),
        'false_selections': int(fp), 'missed_selections': int(fn),
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]}

def choose_threshold(y, scores):
    # Declare a recall floor, not a promised precision score. Avoid a trivial tiny shortlist.
    rows = []
    for threshold in np.round(np.arange(0.50, 0.901, 0.01), 2):
        pred = np.asarray(scores) >= threshold
        tp = int(((np.asarray(y)==1) & pred).sum())
        count = int(pred.sum())
        rows.append({'threshold': float(threshold), 'precision': tp/count if count else 0,
                     'recall': tp/int(np.asarray(y).sum()), 'predicted_selected': count})
    eligible = [row for row in rows if row['recall'] >= 0.5 and row['predicted_selected'] >= 50]
    if not eligible:
        return 0.5, rows, False
    best = max(eligible, key=lambda row: (row['precision'], row['recall'], -row['threshold']))
    return best['threshold'], rows, True

def run():
    details = json.loads((OUT / 'evaluation.json').read_text())
    model = joblib.load(OUT / 'models.joblib')[details['selected_model']]
    data = pd.read_csv(DATA / 'cleaned.csv')
    splits = pd.read_csv(OUT / 'split_manifest.csv')
    training = data[data.candidate_id.isin(splits.loc[splits.split.eq('train'), 'candidate_id'])]
    y = training.selection_status.eq('Selected').astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_predict(model, training[FEATURES], y, cv=cv, method='predict_proba', n_jobs=2)[:, 1]
    threshold, sweep, feasible = choose_threshold(y, scores)
    policy = {'model': details['selected_model'], 'threshold': threshold, 'recall_floor': 0.5,
        'minimum_predicted_selected': 50, 'threshold_grid': '0.50 through 0.90, step 0.01',
        'eligible_policy_found': feasible, 'source': 'Out-of-fold predictions from original training records only',
        'validation_caveat': 'Hyperparameters were selected on these training records; OOF policy results are development estimates, not an unbiased final test.',
        'model_sha256': hashlib.sha256((OUT / 'models.joblib').read_bytes()).hexdigest(),
        'fresh_seed_predeclared': 20261013, 'fresh_corruption_seed_predeclared': 20261014}
    # Freeze and persist before generating or viewing fresh labels.
    (OUT / 'decision_policy.json').write_text(json.dumps(policy, indent=2), encoding='utf-8')
    pd.DataFrame(sweep).to_csv(OUT / 'threshold_validation.csv', index=False)
    fresh = generate(n=9000, seed=policy['fresh_seed_predeclared'])
    fresh['candidate_id'] = [f'SYN-{i:05d}' for i in range(9001, 18001)]
    fresh_raw = corrupt(fresh, seed=policy['fresh_corruption_seed_predeclared'])
    fresh_clean, _, _, counts = clean(fresh_raw)
    assert not set(fresh_clean.candidate_id) & set(data.candidate_id)
    fresh_clean.to_csv(DATA / 'fresh_evaluation.csv', index=False)
    fresh_y = fresh_clean.selection_status.eq('Selected').astype(int)
    fresh_scores = model.predict_proba(fresh_clean[FEATURES])[:, 1]
    evaluation = {'scope': 'Fresh same-simulator evaluation; not external real-world validation',
        'policy': policy, 'fresh_cleaning': counts,
        'default': measure(fresh_y, fresh_scores, 0.5),
        'precision_focused': measure(fresh_y, fresh_scores, threshold),
        'confidence_interval_method': '95% Wilson intervals for binomial proportions; 1000 row-count bootstrap replicates for F1. Conditional on the fitted model; simulator uncertainty is not included.'}
    (OUT / 'reliability.json').write_text(json.dumps(evaluation, indent=2), encoding='utf-8')
    # Separate diagnostic; never changes model, threshold or the headline dataset.
    sensitivity = []
    for noise in [0.0, 0.65, 1.3]:
        scenario = generate(n=9000, seed=20261015, label_noise=noise)
        scenario['candidate_id'] = [f'SYN-{i:05d}' for i in range(18001, 27001)]
        scenario, _, _, _ = clean(corrupt(scenario, seed=20261016))
        result = measure(scenario.selection_status.eq('Selected').astype(int), model.predict_proba(scenario[FEATURES])[:, 1], threshold)
        sensitivity.append({'Hidden noise SD': noise, **{k: result[k] for k in ['accuracy', 'precision', 'recall', 'f1']}})
    pd.DataFrame(sensitivity).to_csv(OUT / 'noise_sensitivity.csv', index=False)
    print(json.dumps(evaluation, indent=2))

if __name__ == '__main__':
    run()
