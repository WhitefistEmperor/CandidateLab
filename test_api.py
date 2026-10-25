import unittest
from app import app
from software_schema import FEATURES

class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        csrf = self.client.get('/api/session').json['csrf']
        login = self.client.post('/api/login', json={'username':'hr','password':'hr123'}, headers={'X-CSRF-Token':csrf})
        self.client.environ_base['HTTP_X_CSRF_TOKEN'] = login.json['csrf']
        self.profile = {field: 7 if field == 'cgpa' else 1 if field == 'internships' else 3 if field == 'projects' else 60 for field in FEATURES}

    def test_frontend_and_schema(self):
        with self.client.get('/') as response:
            self.assertEqual(response.status_code, 200)
        with self.client.get('/assets/app.js') as response:
            self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.client.get('/api/schema').json['features']), 12)

    def test_prediction_and_threshold_consistency(self):
        response = self.client.post('/api/predict', json={'candidates': [self.profile, self.profile]})
        self.assertEqual(response.status_code, 200)
        results = response.json['results']
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0]['outcome'], 'Selected' if results[0]['model_score'] >= results[0]['threshold'] else 'Not Selected')

    def test_invalid_inputs(self):
        for value in [-1, 101, True, None]:
            invalid = dict(self.profile, dsa=value)
            self.assertEqual(self.client.post('/api/predict', json={'candidates': [invalid]}).status_code, 400)
        self.assertEqual(self.client.post('/api/predict', json={'candidates': []}).status_code, 400)

    def test_csv_batch(self):
        import io, pandas as pd
        csv = pd.DataFrame([self.profile, self.profile]).to_csv(index=False).encode()
        response = self.client.post('/api/batch', data={'file': (io.BytesIO(csv), 'candidates.csv')})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json['results']), 2)

if __name__ == '__main__':
    unittest.main()
