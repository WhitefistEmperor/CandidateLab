"""Feature definitions for a synthetic entry-level software-engineer assessment."""
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / 'data/software'
OUT = ROOT / 'results/software'
BACKGROUND = ['cgpa', 'internships', 'projects']
ASSESSMENTS = ['coding_correctness', 'dsa', 'debugging', 'oop', 'dbms',
               'os_networks', 'project_review', 'problem_solving', 'communication']
FEATURES = BACKGROUND + ASSESSMENTS
RANGES = {'cgpa': (0, 10), 'internships': (0, 4), 'projects': (0, 8),
          **{field: (0, 100) for field in ASSESSMENTS}}
LABELS = {field: field.replace('_', ' ').title() for field in FEATURES}
LABELS.update({'cgpa': 'CGPA', 'dsa': 'Data structures & algorithms',
               'oop': 'Object-oriented programming', 'dbms': 'Database fundamentals',
               'os_networks': 'Operating systems & networking'})

def validate_batch(batch):
    """Prediction requires complete, in-range scores; cleaning is a separate process."""
    import pandas as pd
    missing = set(FEATURES) - set(batch.columns)
    if missing:
        raise ValueError(f'Missing input columns: {sorted(missing)}')
    inputs = batch[FEATURES].apply(pd.to_numeric, errors='raise')
    if inputs.empty or inputs.isna().any().any():
        raise ValueError('Provide at least one complete assessment row. Incomplete candidates need review.')
    for field, (low, high) in RANGES.items():
        if not inputs[field].between(low, high).all():
            raise ValueError(f'{LABELS[field]} must be between {low} and {high}.')
        if field in ['internships', 'projects'] and inputs[field].mod(1).ne(0).any():
            raise ValueError(f'{LABELS[field]} must be a whole number.')
    return inputs
