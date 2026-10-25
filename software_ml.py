"""Training and diagnostics for a software-engineering simulation."""
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance
from software_schema import DATA, OUT, FEATURES, BACKGROUND, ASSESSMENTS

def pipeline(estimator):
    return Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler()), ('model', estimator)])

def train(data):
    X = data[FEATURES]
    y = data.selection_status.map({'Selected': 1, 'Not Selected': 0})
    indices = np.arange(len(data))
    tr, te = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
    Xt, Xh, yt, yh = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    assert set(data.iloc[tr].candidate_id).isdisjoint(set(data.iloc[te].candidate_id))
    cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(Xt, yt))
    specifications = {
        'Logistic Regression': (LogisticRegression(max_iter=2000, random_state=42), {'model__C': [0.01, 0.1, 1, 10]}),
        'Decision Tree': (DecisionTreeClassifier(random_state=42), {'model__max_depth': [3, 5, 7], 'model__min_samples_leaf': [20, 50]}),
        'Random Forest': (RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=1), {'model__max_depth': [5, 8], 'model__min_samples_leaf': [10, 30]})}
    models, rows, parameters = {}, [], {}
    for name, (estimator, grid) in specifications.items():
        search = GridSearchCV(pipeline(estimator), grid, scoring='f1_macro', cv=cv, n_jobs=2)
        search.fit(Xt, yt)
        models[name] = search.best_estimator_
        parameters[name] = search.best_params_
        rows.append({'Model': name, 'Train macro F1': f1_score(yt, search.predict(Xt), average='macro'),
                     'CV macro F1': search.best_score_,
                     'CV SD': search.cv_results_['std_test_score'][search.best_index_]})
    selected = max(rows, key=lambda row: row['CV macro F1'])['Model']
    matrices = {}
    for row in rows:
        prediction = models[row['Model']].predict(Xh)
        row.update({'Accuracy': accuracy_score(yh, prediction), 'Precision': precision_score(yh, prediction),
                    'Recall': recall_score(yh, prediction), 'F1': f1_score(yh, prediction),
                    'Test macro F1': f1_score(yh, prediction, average='macro'),
                    'Balanced accuracy': balanced_accuracy_score(yh, prediction)})
        matrices[row['Model']] = confusion_matrix(yh, prediction, labels=[0, 1]).tolist()
    ablations = []
    for label, columns in [('Background only', BACKGROUND), ('Assessment only', ASSESSMENTS), ('Combined', FEATURES)]:
        search = GridSearchCV(pipeline(LogisticRegression(max_iter=2000, random_state=42)),
            {'model__C': [0.01, 0.1, 1, 10]}, scoring='f1_macro', cv=cv, n_jobs=2)
        search.fit(Xt[columns], yt)
        p = search.predict(Xh[columns])
        ablations.append({'Inputs': label, 'CV macro F1': search.best_score_,
                          'Test macro F1': f1_score(yh, p, average='macro'), 'Accuracy': accuracy_score(yh, p)})
    chosen = models[selected]
    sizes, training, validation = learning_curve(chosen, Xt, yt, cv=cv,
        train_sizes=[0.25, 0.5, 0.75, 1], scoring='f1_macro', n_jobs=2, shuffle=True, random_state=42)
    curves = pd.DataFrame({'Training rows': sizes, 'Training macro F1': training.mean(axis=1),
        'Validation macro F1': validation.mean(axis=1), 'Validation SD': validation.std(axis=1)})
    importance = permutation_importance(chosen, Xh, yh, scoring='f1_macro', n_repeats=5, random_state=42, n_jobs=2)
    importance_frame = pd.DataFrame({'Feature': FEATURES, 'Macro F1 decrease': importance.importances_mean,
        'SD': importance.importances_std}).sort_values('Macro F1 decrease', ascending=False)
    prediction = chosen.predict(Xh)
    errors = data.iloc[te].copy()
    errors['Predicted'] = np.where(prediction == 1, 'Selected', 'Not Selected')
    errors = errors[errors.selection_status.ne(errors.Predicted)]
    baseline = DummyClassifier(strategy='most_frequent').fit(Xt, yt)
    details = {'dataset_type': 'Synthetic simulation; not evidence of real hiring accuracy',
        'selected_model': selected, 'training_rows': len(Xt), 'test_rows': len(Xh),
        'train_selected_fraction': float(yt.mean()), 'test_selected_fraction': float(yh.mean()),
        'candidate_id_overlap': 0, 'parameters': parameters, 'confusion_matrices': matrices,
        'baseline_accuracy': accuracy_score(yh, baseline.predict(Xh)), 'information_comparison': ablations,
        'features': FEATURES, 'split_seed': 42, 'model_selection': 'Training CV macro F1 only'}
    splits = pd.DataFrame({'candidate_id': data.candidate_id, 'split': 'train'})
    splits.loc[te, 'split'] = 'test'
    return models, pd.DataFrame(rows), details, curves, importance_frame, errors, splits

if __name__ == '__main__':
    data = pd.read_csv(DATA / 'cleaned.csv')
    models, metrics, details, curves, importance, errors, splits = train(data)
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, frame in [('metrics', metrics), ('learning_curve', curves), ('importance', importance), ('errors', errors), ('split_manifest', splits)]:
        frame.to_csv(OUT / f'{filename}.csv', index=False)
    # Publish the manifest last so the app can detect a complete training run.
    joblib.dump(models, OUT / 'models.joblib')
    (OUT / 'evaluation.json').write_text(json.dumps(details, indent=2), encoding='utf-8')
    print(metrics.to_string(index=False))
    print(json.dumps(details, indent=2))
