"""A transparent simulation, NOT observed hiring data or manually collected records."""
import json
import hashlib
import numpy as np
import pandas as pd
from software_schema import DATA, FEATURES, ASSESSMENTS

def generate(n=9000, seed=42, label_noise=0.65):
    rng = np.random.default_rng(seed)
    # Fictional correlated abilities. No empirical population calibration is claimed.
    aptitude = rng.normal(0, 1, n)
    practice = 0.45 * aptitude + rng.normal(0, 0.9, n)
    fundamentals = 0.5 * aptitude + rng.normal(0, 0.85, n)
    communication = rng.normal(0, 1, n)
    data = pd.DataFrame({'candidate_id': [f'SYN-{i:05d}' for i in range(1, n + 1)],
        'cgpa': np.round(np.clip(7 + 0.65 * fundamentals + rng.normal(0, 0.5, n), 0, 10), 2),
        'internships': np.clip(rng.poisson(np.exp(0.2 * practice)), 0, 4),
        'projects': np.clip(rng.poisson(np.exp(0.9 + 0.3 * practice)), 0, 8)})
    abilities = {'coding_correctness': 0.7 * practice + 0.3 * aptitude,
        'dsa': 0.6 * aptitude + 0.4 * fundamentals,
        'debugging': 0.65 * practice + 0.35 * aptitude,
        'oop': fundamentals, 'dbms': fundamentals,
        'os_networks': fundamentals, 'project_review': practice,
        'problem_solving': aptitude, 'communication': communication}
    for field in ASSESSMENTS:
        data[field] = np.round(np.clip(60 + 16 * abilities[field] + rng.normal(0, 9, n), 0, 100)).astype(int)
    # An explicitly invented outcome process. Hidden noise represents unmeasured
    # factors in this simulation; it is not evidence about real employers.
    z = (data[ASSESSMENTS] - 60) / 20
    logit = (-0.45 + 0.8 * z.coding_correctness + 0.6 * z.dsa + 0.35 * z.debugging
             + 0.25 * z.project_review + 0.3 * z.problem_solving + 0.25 * z.communication
             + 0.15 * z.oop + 0.1 * z.dbms + 0.1 * z.os_networks
             + 0.1 * (data.cgpa - 7) + 0.12 * (data.internships - 1)
             + 0.05 * (data.projects - 3)
             + 0.3 * np.tanh(z.coding_correctness * z.dsa)
             - 0.7 * (data.coding_correctness < 35)
             + rng.normal(0, label_noise, n))
    probability = 1 / (1 + np.exp(-logit))
    data['selection_status'] = np.where(rng.random(n) < probability, 'Selected', 'Not Selected')
    return data

def corrupt(reference, seed=314):
    """Separate labelled exercise copy with known, artificial data-entry defects."""
    rng = np.random.default_rng(seed)
    raw = reference.copy()
    for field in FEATURES:
        raw[field] = raw[field].astype(float)
        raw.loc[rng.choice(len(raw), size=180, replace=False), field] = np.nan
    raw.loc[rng.choice(len(raw), size=90, replace=False), 'dsa'] = 140
    raw.loc[rng.choice(len(raw), size=45, replace=False), 'internships'] = -2
    raw.loc[rng.choice(len(raw), size=20, replace=False), FEATURES] = np.nan
    whitespace = rng.choice(len(raw), size=300, replace=False)
    raw.loc[whitespace, 'selection_status'] = ' ' + raw.loc[whitespace, 'selection_status'] + ' '
    synonyms = rng.choice(len(raw), size=150, replace=False)
    raw.loc[synonyms, 'selection_status'] = raw.loc[synonyms, 'selection_status'].str.strip().str.lower()
    duplicates = raw.iloc[rng.choice(len(raw), size=200, replace=False)]
    return pd.concat([raw, duplicates], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

if __name__ == '__main__':
    DATA.mkdir(parents=True, exist_ok=True)
    reference = generate()
    raw = corrupt(reference)
    reference.to_csv(DATA / 'synthetic_reference.csv', index=False)
    raw.to_csv(DATA / 'raw_entry_exercise.csv', index=False)
    metadata = {'kind': 'synthetic demonstration, not real hiring records',
        'generation_seed': 42, 'corruption_seed': 314, 'generated_candidates': len(reference),
        'raw_exercise_rows_including_duplicates': len(raw),
        'features': FEATURES, 'source': 'generate_software_data.py in this project',
        'assumptions': 'Correlated fictional abilities, noisy observed scores, stochastic logistic selection with an interaction and low-coding penalty. All coefficients are invented, not empirically validated.',
        'quality_exercise': '180 missing entries per feature, 90 out-of-range DSA entries, 45 invalid internship counts, 20 extensively incomplete rows, label whitespace/case variants, 200 duplicate copies. Defects can overlap.',
        'reference_sha256': hashlib.sha256((DATA / 'synthetic_reference.csv').read_bytes()).hexdigest()}
    (DATA / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Generated {len(reference):,} synthetic candidates and {len(raw):,} raw exercise rows.')
