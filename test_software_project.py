"""Meaningful data-cleaning, leakage and inference checks. Run with unittest."""
import unittest
import joblib
import numpy as np
import pandas as pd
from software_schema import DATA, OUT, FEATURES, validate_batch
from generate_software_data import generate
from clean_software_data import clean
from interview_explain import forest_contributions
from software_reliability import choose_threshold, wilson, measure

class SoftwareProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(DATA / 'raw_entry_exercise.csv')
        cls.cleaned = pd.read_csv(DATA / 'cleaned.csv')
        cls.models = joblib.load(OUT / 'models.joblib')

    def test_cleaning_accounting_and_preserves_missingness(self):
        cleaned, review, report, stats = clean(self.raw)
        self.assertEqual(stats['raw_rows'], len(cleaned) + len(review) + stats['exact_duplicate_rows_removed'])
        self.assertEqual(cleaned.candidate_id.nunique(), len(cleaned))
        self.assertEqual(stats['exact_duplicate_rows_removed'], 200)
        self.assertGreater(cleaned[FEATURES].isna().sum().sum(), 0)
        self.assertTrue(report['Invalid range/type'].sum() > 0)
        self.assertFalse(set(cleaned.candidate_id) & set(review.candidate_id))

    def test_bad_entries_and_conflicting_ids_go_to_review(self):
        d = generate(4)
        d['internships'] = d.internships.astype(float)
        d.loc[0, 'dsa'] = 120
        d.loc[1, 'selection_status'] = 'unknown'
        d.loc[2, 'internships'] = 1.5
        d = pd.concat([d, d.iloc[[3]].assign(dsa=99)], ignore_index=True)
        cleaned, review, report, stats = clean(d)
        self.assertTrue(pd.isna(cleaned.loc[cleaned.candidate_id.eq('SYN-00001'), 'dsa'].iloc[0]))
        self.assertTrue(pd.isna(cleaned.loc[cleaned.candidate_id.eq('SYN-00003'), 'internships'].iloc[0]))
        self.assertEqual(len(review), 3)

    def test_generation_is_repeatable(self):
        pd.testing.assert_frame_equal(generate(30, seed=42), generate(30, seed=42))
        reference = pd.read_csv(DATA / 'synthetic_reference.csv')
        pd.testing.assert_frame_equal(reference, generate(), check_dtype=False)

    def test_training_medians_do_not_use_test_records(self):
        splits = pd.read_csv(OUT / 'split_manifest.csv')
        training_ids = set(splits.loc[splits.split.eq('train'), 'candidate_id'])
        test_ids = set(splits.loc[splits.split.eq('test'), 'candidate_id'])
        self.assertFalse(training_ids & test_ids)
        self.assertEqual(len(training_ids) + len(test_ids), len(self.cleaned))
        training = self.cleaned.loc[self.cleaned.candidate_id.isin(training_ids), FEATURES]
        expected = np.nanmedian(training.to_numpy(), axis=0)
        for model in self.models.values():
            np.testing.assert_allclose(model.named_steps['impute'].statistics_, expected)

    def test_batch_validation_and_predictions(self):
        batch = generate(5)[FEATURES]
        inputs = validate_batch(batch)
        for model in self.models.values():
            self.assertEqual(len(model.predict(inputs)), 5)
        for field, value in [('coding_correctness', 101), ('internships', 1.5), ('cgpa', np.nan)]:
            invalid = batch.astype(float).copy()
            invalid.loc[0, field] = value
            with self.assertRaises(ValueError):
                validate_batch(invalid)
        with self.assertRaises(ValueError):
            validate_batch(batch.drop(columns='dsa'))

    def test_forest_explanation_reconstructs_model_score(self):
        model = self.models['Random Forest']
        for _, row in self.cleaned.sample(5, random_state=3)[FEATURES].iterrows():
            frame = row.to_frame().T
            baseline, contributions = forest_contributions(model, frame)
            self.assertAlmostEqual(baseline + contributions.sum(), model.predict_proba(frame)[0, 1], places=7)

    def test_precision_policy_requires_recall_and_nontrivial_support(self):
        y = np.array([1]*100 + [0]*100)
        scores = np.r_[np.repeat(0.8, 60), np.repeat(0.55, 40), np.repeat(0.6, 100)]
        threshold, rows, feasible = choose_threshold(y, scores)
        chosen = next(row for row in rows if row['threshold'] == threshold)
        self.assertTrue(feasible)
        self.assertGreaterEqual(chosen['recall'], 0.5)
        self.assertGreaterEqual(chosen['predicted_selected'], 50)
        result = measure(y, scores, threshold)
        self.assertEqual(result['precision'], 1)
        self.assertAlmostEqual(result['recall'], 0.6)
        self.assertTrue(0 <= result['precision_95_ci'][0] <= 1)
        self.assertEqual(wilson(0, 0), [None, None])

    def test_fresh_batch_is_separate_and_policy_matches_model(self):
        import hashlib, json
        fresh = pd.read_csv(DATA / 'fresh_evaluation.csv')
        policy = json.loads((OUT / 'decision_policy.json').read_text())
        self.assertFalse(set(fresh.candidate_id) & set(self.cleaned.candidate_id))
        self.assertEqual(policy['model_sha256'], hashlib.sha256((OUT / 'models.joblib').read_bytes()).hexdigest())

if __name__ == '__main__':
    unittest.main()
