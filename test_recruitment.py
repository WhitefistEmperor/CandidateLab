"""Check the real workflow, permissions and file storage without AWS charges."""
import io
from contextlib import closing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app import app
from recruitment import database
from software_schema import FEATURES


class RecruitmentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.old_config = {key:app.config[key] for key in ['DATABASE','DATA_DIR','S3_BUCKET']}
        self.directory_path = Path(self.directory.name)
        (self.directory_path / 'resumes').mkdir()
        # Use an isolated copy of the empty schema, never modify saved app records.
        import sqlite3
        with closing(sqlite3.connect(app.config['DATABASE'])) as source:
            schema = source.execute("SELECT sql FROM sqlite_master WHERE type='table'").fetchall()
        app.config.update(DATABASE=str(self.directory_path / 'test.db'), DATA_DIR=self.directory_path, S3_BUCKET='')
        with app.app_context():
            for (sql,) in schema:
                database().execute(sql)
            database().commit()
        self.client = app.test_client()
        self.login('hr')

    def tearDown(self):
        app.config.update(self.old_config)
        self.directory.cleanup()

    def login(self, username):
        csrf = self.client.get('/api/session').json['csrf']
        response = self.client.post('/api/login', json={'username':username,'password':username+'123'}, headers={'X-CSRF-Token':csrf})
        self.assertEqual(response.status_code, 200)
        self.client.environ_base['HTTP_X_CSRF_TOKEN'] = response.json['csrf']

    def candidate(self, skills='python, sql, flask'):
        response = self.client.post('/api/candidates', json={'name':'Asha Mehta','email':'asha@example.com','skills':skills})
        self.assertEqual(response.status_code, 201)
        return response.json['id']

    def test_hr_manager_workflow_and_persistence(self):
        candidate_id = self.candidate()
        scores = {f:8 if f=='cgpa' else 1 if f=='internships' else 3 if f=='projects' else 75 for f in FEATURES}
        response = self.client.post(f'/api/candidates/{candidate_id}/assessment', json={'scores':scores})
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.json['predicted_selection'], ['Selected','Not Selected'])
        self.assertEqual(response.json['predicted_selection'], response.json['outcome'])
        self.assertEqual(response.json['recommendation'], response.json['outcome'])
        record = self.client.get('/api/candidates').json['candidates'][0]
        self.assertEqual(record['status'], 'Applied')
        self.assertEqual(record['scores'], scores)
        self.assertEqual(record['predicted_selection'], response.json['outcome'])
        self.client.patch(f'/api/candidates/{candidate_id}/status', json={'status':'Interview'})
        self.login('manager')
        self.assertNotIn('predicted_selection', self.client.get('/api/candidates').json['candidates'][0])
        self.assertEqual(self.client.post('/api/candidates', json={}).status_code, 403)
        self.assertEqual(self.client.post('/api/predict', json={'candidates':[scores]}).status_code, 403)
        self.assertEqual(self.client.patch(f'/api/candidates/{candidate_id}/status', json={'status':'Selected'}).status_code, 403)
        self.assertEqual(self.client.get(f'/api/candidates/{candidate_id}/resume').status_code, 403)
        manager_record = self.client.get('/api/candidates').json['candidates'][0]
        for field in ['email','scores','model_score','resume_name']:
            self.assertNotIn(field, manager_record)
        project_id = self.client.post('/api/projects', json={'name':'Inventory portal','skills':'Python, SQL, sql'}).json['id']
        match = self.client.get(f'/api/projects/{project_id}/matches').json['candidates'][0]
        self.assertEqual(match['match_percent'],100)
        self.assertEqual(match['matched_skills'],['python','sql'])
        for _ in range(2):
            self.assertEqual(self.client.post(f'/api/projects/{project_id}/shortlist', json={'candidate_id':candidate_id}).status_code,200)
        # New client session verifies records outlive a login and DB connection.
        self.client = app.test_client()
        self.login('hr')
        self.assertTrue(self.client.get(f'/api/projects/{project_id}/matches').json['candidates'][0]['shortlisted'])
        self.assertEqual(self.client.post('/api/projects',json={'name':'Other','skills':'sql'}).status_code,403)
        self.login('manager')
        self.client.delete(f'/api/projects/{project_id}/shortlist/{candidate_id}')
        self.assertFalse(self.client.get(f'/api/projects/{project_id}/matches').json['candidates'][0]['shortlisted'])

    def test_invalid_data_and_authentication(self):
        self.assertEqual(self.client.post('/api/candidates',json={'name':'','email':'x','skills':'python'}).status_code,400)
        candidate_id = self.candidate()
        self.assertEqual(self.client.patch(f'/api/candidates/{candidate_id}/status',json={'status':'Unknown'}).status_code,400)
        self.assertEqual(self.client.post(f'/api/candidates/{candidate_id}/assessment',json={'scores':{'cgpa':8}}).status_code,400)
        self.assertEqual(self.client.post('/api/candidates',json=[],headers={'X-CSRF-Token':'bad'}).status_code,403)
        self.assertEqual(self.client.post('/api/candidates',json={},headers={'Origin':'https://other.example'}).status_code,403)
        self.client.post('/api/logout',json={})
        self.assertEqual(self.client.get('/api/candidates').status_code,401)
        token = self.client.get('/api/session').json['csrf']
        self.assertEqual(self.client.post('/api/login',json={'username':'hr','password':'wrong'},headers={'X-CSRF-Token':token}).status_code,401)

    def test_local_pdf_upload_replace_and_download(self):
        candidate_id = self.candidate()
        url = f'/api/candidates/{candidate_id}/resume'
        self.assertEqual(self.client.post(url,data={'file':(io.BytesIO(b'not pdf'),'resume.pdf')}).status_code,400)
        pdf = b'%PDF-1.4\nSample resume\n%%EOF'
        self.assertEqual(self.client.post(url,data={'file':(io.BytesIO(pdf),'resume.pdf')}).status_code,200)
        response = self.client.get(url)
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.data,pdf)
        response.close()
        replacement = b'%PDF-1.4 replacement'
        self.assertEqual(self.client.post(url,data={'file':(io.BytesIO(replacement),'new.pdf')}).status_code,200)
        self.assertEqual(len(list((self.directory_path / 'resumes').glob('*.pdf'))),1)
        with self.client.get(url) as download:
            self.assertEqual(download.data,replacement)

    def test_s3_upload_download_and_failure(self):
        candidate_id = self.candidate()
        app.config['S3_BUCKET'] = 'test-private-bucket'
        url = f'/api/candidates/{candidate_id}/resume'
        with patch('recruitment.s3_client') as factory:
            factory.return_value.generate_presigned_url.return_value = 'https://example.com/signed'
            self.assertEqual(self.client.post(url,data={'file':(io.BytesIO(b'%PDF-1.4 test'),'resume.pdf')}).status_code,200)
            call = factory.return_value.upload_fileobj.call_args
            self.assertEqual(call.args[1],'test-private-bucket')
            self.assertTrue(call.args[2].startswith('resumes/'))
            self.assertEqual(self.client.get(url).status_code,302)
            self.assertEqual(factory.return_value.generate_presigned_url.call_args.kwargs['ExpiresIn'],300)
            factory.return_value.upload_fileobj.side_effect = RuntimeError('offline')
            self.assertEqual(self.client.post(url,data={'file':(io.BytesIO(b'%PDF-1.4 new'),'new.pdf')}).status_code,503)
            self.assertTrue(self.client.get('/api/candidates').json['candidates'][0]['has_resume'])


if __name__ == '__main__':
    unittest.main()
