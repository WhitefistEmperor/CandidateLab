"""Validate entry errors without using the pristine simulation to fill missing scores."""
import json
import numpy as np
import pandas as pd
from software_schema import DATA, FEATURES, RANGES

def clean(raw):
    d = raw.copy()
    required = set(FEATURES + ['candidate_id', 'selection_status'])
    if not required.issubset(d.columns):
        raise ValueError(f'Missing columns: {sorted(required-set(d.columns))}')
    original_labels = d.selection_status.astype('str')
    normalized = original_labels.str.strip().str.lower().map({'selected': 'Selected', 'not selected': 'Not Selected'})
    d['selection_status'] = normalized
    d['candidate_id'] = d.candidate_id.astype('str').str.strip()
    numeric_report = []
    for field in FEATURES:
        original_missing = int(d[field].isna().sum())
        numbers = pd.to_numeric(d[field], errors='coerce')
        low, high = RANGES[field]
        invalid = numbers.notna() & (~np.isfinite(numbers) | numbers.lt(low) | numbers.gt(high))
        if field in ['internships', 'projects']:
            invalid |= numbers.notna() & numbers.mod(1).ne(0)
        parsed_failures = int((d[field].notna() & numbers.isna()).sum())
        d[field] = numbers.mask(invalid)
        numeric_report.append({'Feature': field, 'Original missing': original_missing,
            'Non-numeric entries': parsed_failures, 'Invalid range/type': int(invalid.sum()),
            'Missing after validation': int(d[field].isna().sum())})
    # IDs are used for de-duplication only, never as predictive features.
    duplicate = d.duplicated(subset=['candidate_id'] + FEATURES + ['selection_status'])
    duplicates_removed = int(duplicate.sum())
    d = d.loc[~duplicate].copy()
    # Conflicting records with one ID require review instead of choosing one.
    repeated_id = d.candidate_id.duplicated(keep=False)
    label_missing = d.selection_status.isna()
    id_invalid = ~d.candidate_id.str.fullmatch(r'SYN-\d{5}')
    excessive_missing = d[FEATURES].isna().mean(axis=1).gt(0.4)
    reasons = pd.Series('', index=d.index)
    for mask, reason in [(repeated_id, 'Conflicting repeated ID; '), (label_missing, 'Unknown label; '),
                         (id_invalid, 'Invalid ID; '), (excessive_missing, 'More than 40% inputs missing; ')]:
        reasons.loc[mask] += reason
    review = d.loc[reasons.ne('')].copy()
    review['review_reason'] = reasons.loc[reasons.ne('')]
    cleaned = d.loc[reasons.eq(''), ['candidate_id'] + FEATURES + ['selection_status']].copy()
    stats = {'dataset_type': 'synthetic data-entry cleaning exercise', 'raw_rows': len(raw),
             'label_cells_standardized': int(normalized.notna().mul(normalized.ne(original_labels)).sum()),
             'exact_duplicate_rows_removed': duplicates_removed, 'review_rows': len(review),
             'cleaned_rows': len(cleaned), 'remaining_missing_cells': int(cleaned[FEATURES].isna().sum().sum()),
             'imputation_policy': 'No global filling. Median imputation is fitted inside each training fold only.',
             'reference_policy': 'synthetic_reference.csv is not read during cleaning or training.'}
    return cleaned, review, pd.DataFrame(numeric_report), stats

if __name__ == '__main__':
    raw = pd.read_csv(DATA / 'raw_entry_exercise.csv')
    cleaned, review, report, stats = clean(raw)
    cleaned.to_csv(DATA / 'cleaned.csv', index=False)
    review.to_csv(DATA / 'review_queue.csv', index=False)
    report.to_csv(DATA / 'cleaning_by_feature.csv', index=False)
    (DATA / 'cleaning_summary.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(json.dumps(stats, indent=2))
