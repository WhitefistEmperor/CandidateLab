"""Rebuild the reproducible synthetic demonstration in the teaching order."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).parent
steps = [('Generate synthetic data and entry-error exercise', 'generate_software_data.py'),
         ('Validate, clean and export review records', 'clean_software_data.py'),
         ('Train and evaluate classifiers', 'software_ml.py'),
         ('Freeze threshold and evaluate fresh synthetic candidates', 'software_reliability.py'),
         ('Run integrity and leakage checks', 'test_software_project.py')]
if __name__ == '__main__':
    for number, (description, script) in enumerate(steps, 1):
        print(f'\nStep {number}: {description}', flush=True)
        subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)
    print('\nComplete. Start the interface: python app.py')
