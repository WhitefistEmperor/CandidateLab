"""Small recruitment workflow: two roles, SQLite records and optional S3 files."""
import hmac
import json
import logging
import os
from pathlib import Path
import secrets
import sqlite3
from functools import wraps
from datetime import timedelta

from flask import current_app, g, jsonify, request, session, send_file, redirect
from werkzeug.utils import secure_filename
from logging.handlers import RotatingFileHandler

STATUSES = ['Applied', 'Interview', 'Selected', 'Rejected']


def database():
    if 'recruitment_db' not in g:
        g.recruitment_db = sqlite3.connect(current_app.config['DATABASE'])
        g.recruitment_db.row_factory = sqlite3.Row
        g.recruitment_db.execute('PRAGMA foreign_keys = ON')
    return g.recruitment_db


def permitted(*roles):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            if session.get('role') not in roles:
                return jsonify(error='Your account cannot perform this action.'), 403
            return function(*args, **kwargs)
        return wrapped
    return decorate


def text_field(payload, key, maximum=120):
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > maximum:
        raise ValueError(f'Provide {key.replace("_", " ")} (maximum {maximum} characters).')
    return value.strip()


def skill_list(payload):
    value = text_field(payload, 'skills', 300)
    skills = list(dict.fromkeys(s.strip().lower() for s in value.split(',') if s.strip()))
    if not skills:
        raise ValueError('Enter at least one skill, separated by commas.')
    return ', '.join(skills)


def s3_client():
    import boto3
    from botocore.config import Config
    region = os.environ.get('AWS_DEFAULT_REGION', 'ap-south-1')
    # Sign for the regional endpoint so downloads do not redirect to another host.
    return boto3.client('s3', region_name=region,
        endpoint_url=f'https://s3.{region}.amazonaws.com',
        config=Config(signature_version='s3v4', s3={'addressing_style': 'virtual'}))


def register_recruitment(app, assess):
    directory = Path(os.environ.get('DATA_DIR', str(Path(__file__).parent / 'instance'))).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'resumes').mkdir(exist_ok=True)
    secret_file = directory / '.session-key'
    if not os.environ.get('SECRET_KEY') and not secret_file.exists():
        secret_file.write_text(secrets.token_hex(32))
    app.config.update(
        SECRET_KEY=os.environ.get('SECRET_KEY') or secret_file.read_text().strip(),
        DATABASE=str(directory / 'recruitment.db'), DATA_DIR=directory,
        S3_BUCKET=os.environ.get('S3_BUCKET', ''), MAX_CONTENT_LENGTH=5 * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax',
        SESSION_COOKIE_SECURE=os.environ.get('COOKIE_SECURE') == '1',
        PERMANENT_SESSION_LIFETIME=timedelta(hours=4),
    )
    handler = RotatingFileHandler(directory / 'application.log', maxBytes=1_000_000, backupCount=2)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)

    @app.teardown_appcontext
    def close_database(error=None):
        connection = g.pop('recruitment_db', None)
        if connection is not None:
            connection.close()

    with app.app_context():
        database().executescript('''
            CREATE TABLE IF NOT EXISTS candidates (
                id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL,
                skills TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'Applied',
                scores TEXT, model_score REAL, recommendation TEXT,
                resume_key TEXT, resume_name TEXT, resume_storage TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY, name TEXT NOT NULL, skills TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS shortlists (
                project_id INTEGER REFERENCES projects(id),
                candidate_id INTEGER REFERENCES candidates(id),
                PRIMARY KEY (project_id, candidate_id)
            );
        ''')
        database().commit()

    @app.before_request
    def protect_api():
        if not request.path.startswith('/api/'):
            return
        if request.path not in ['/api/login', '/api/session'] and not session.get('role'):
            return jsonify(error='Please log in first.'), 401
        if request.method in ['POST', 'PATCH', 'DELETE']:
            origin = request.headers.get('Origin')
            if origin and origin.rstrip('/') != request.host_url.rstrip('/'):
                return jsonify(error='Requests must come from this application.'), 403
            token = request.headers.get('X-CSRF-Token', '')
            if not session.get('csrf') or not hmac.compare_digest(token, session['csrf']):
                return jsonify(error='Refresh the page and try again.'), 403
        if request.path in ['/api/predict', '/api/batch'] and session.get('role') != 'hr':
            return jsonify(error='Only HR can submit assessments.'), 403

    @app.after_request
    def log_request(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        if request.path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-store'
            app.logger.info('%s %s status=%s role=%s', request.method, request.path,
                            response.status_code, session.get('role', 'guest'))
        return response

    @app.get('/api/session')
    def session_info():
        if 'csrf' not in session:
            session['csrf'] = secrets.token_hex(24)
        return jsonify(role=session.get('role'), csrf=session['csrf'],
                       storage='Amazon S3' if app.config['S3_BUCKET'] else 'Local files')

    @app.post('/api/login')
    def login():
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify(error='Enter your username and password.'), 400
        username = payload.get('username')
        accounts = {'hr': ('hr', os.environ.get('HR_PASSWORD', 'hr123')),
                    'manager': ('manager', os.environ.get('MANAGER_PASSWORD', 'manager123'))}
        account = accounts.get(username) if isinstance(username, str) else None
        password = payload.get('password')
        if not account or not isinstance(password, str) or not hmac.compare_digest(password.encode(), account[1].encode()):
            return jsonify(error='Incorrect username or password.'), 401
        session.clear()
        session.update(role=account[0], csrf=secrets.token_hex(24))
        session.permanent = True
        return jsonify(role=account[0], csrf=session['csrf'])

    @app.post('/api/logout')
    def logout():
        session.clear()
        return jsonify(ok=True)

    @app.get('/api/candidates')
    def candidates():
        rows = database().execute('SELECT * FROM candidates ORDER BY id DESC').fetchall()
        result = []
        for row in rows:
            candidate = dict(row)
            candidate['has_resume'] = bool(candidate.pop('resume_key'))
            candidate.pop('resume_storage')
            candidate['scores'] = json.loads(candidate['scores']) if candidate['scores'] else None
            # Display historical assessments using the same class names as new ones.
            candidate['recommendation'] = {'Recommended': 'Selected', 'Needs review': 'Not Selected'}.get(
                candidate['recommendation'], candidate['recommendation'])
            if session['role'] == 'hr':
                candidate['predicted_selection'] = candidate['recommendation']
            if session['role'] == 'manager':
                candidate.pop('email')
                candidate.pop('scores')
                candidate.pop('model_score')
                candidate.pop('recommendation')
                candidate.pop('resume_name')
                candidate.pop('has_resume')
            result.append(candidate)
        return jsonify(candidates=result)

    @app.post('/api/candidates')
    @permitted('hr')
    def add_candidate():
        payload = request.get_json(silent=True)
        try:
            if not isinstance(payload, dict):
                raise ValueError('Provide candidate details.')
            name = text_field(payload, 'name')
            email = text_field(payload, 'email', 254)
            if email.count('@') != 1 or any(c.isspace() for c in email) or '.' not in email.split('@')[1] or not email.split('@')[0]:
                raise ValueError('Enter a valid email address.')
            skills = skill_list(payload)
        except ValueError as error:
            return jsonify(error=str(error)), 400
        cursor = database().execute('INSERT INTO candidates (name,email,skills) VALUES (?,?,?)', (name,email,skills))
        database().commit()
        return jsonify(id=cursor.lastrowid), 201

    @app.patch('/api/candidates/<int:candidate_id>/status')
    @permitted('hr')
    def update_status(candidate_id):
        payload = request.get_json(silent=True)
        status = payload.get('status') if isinstance(payload, dict) else None
        if status not in STATUSES:
            return jsonify(error='Choose Applied, Interview, Selected or Rejected.'), 400
        cursor = database().execute('UPDATE candidates SET status=? WHERE id=?', (status,candidate_id))
        database().commit()
        if not cursor.rowcount:
            return jsonify(error='Candidate not found.'), 404
        return jsonify(ok=True)

    @app.post('/api/candidates/<int:candidate_id>/assessment')
    @permitted('hr')
    def save_assessment(candidate_id):
        if not database().execute('SELECT id FROM candidates WHERE id=?', (candidate_id,)).fetchone():
            return jsonify(error='Candidate not found.'), 404
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get('scores'), dict):
            return jsonify(error='Provide all assessment scores.'), 400
        response = assess([payload['scores']])
        if isinstance(response, tuple):
            return response
        result = response.get_json()['results'][0]
        recommendation = result['outcome']
        database().execute('UPDATE candidates SET scores=?,model_score=?,recommendation=? WHERE id=?',
                           (json.dumps(payload['scores']),result['model_score'],recommendation,candidate_id))
        database().commit()
        return jsonify(**result, recommendation=recommendation, predicted_selection=result['outcome'])

    @app.post('/api/candidates/<int:candidate_id>/resume')
    @permitted('hr')
    def upload_resume(candidate_id):
        candidate = database().execute('SELECT id,resume_key,resume_storage FROM candidates WHERE id=?', (candidate_id,)).fetchone()
        if not candidate:
            return jsonify(error='Candidate not found.'), 404
        upload = request.files.get('file')
        if not upload or not upload.filename:
            return jsonify(error='Choose a PDF résumé.'), 400
        filename = secure_filename(upload.filename)
        if not filename.lower().endswith('.pdf') or upload.stream.read(5) != b'%PDF-':
            return jsonify(error='Upload a PDF file, up to 5 MB.'), 400
        upload.stream.seek(0)
        key = f'resumes/{candidate_id}-{secrets.token_hex(8)}.pdf'
        storage = 's3' if app.config['S3_BUCKET'] else 'local'
        try:
            if storage == 's3':
                s3_client().upload_fileobj(upload.stream, app.config['S3_BUCKET'], key,
                    ExtraArgs={'ContentType':'application/pdf', 'ServerSideEncryption':'AES256'})
            else:
                upload.save(current_app.config['DATA_DIR'] / key)
        except Exception:
            app.logger.exception('Resume upload failed')
            return jsonify(error='Storage is unavailable. Check its configuration and try again.'), 503
        database().execute('UPDATE candidates SET resume_key=?,resume_name=?,resume_storage=? WHERE id=?',
                           (key,filename,storage,candidate_id))
        database().commit()
        # Remove the previous document only after the new upload succeeds.
        if candidate['resume_key']:
            try:
                if candidate['resume_storage'] == 's3':
                    s3_client().delete_object(Bucket=app.config['S3_BUCKET'], Key=candidate['resume_key'])
                else:
                    (current_app.config['DATA_DIR'] / candidate['resume_key']).unlink(missing_ok=True)
            except Exception:
                app.logger.exception('Could not remove previous resume')
        return jsonify(ok=True, storage=storage)

    @app.get('/api/candidates/<int:candidate_id>/resume')
    @permitted('hr')
    def download_resume(candidate_id):
        candidate = database().execute('SELECT * FROM candidates WHERE id=?', (candidate_id,)).fetchone()
        if not candidate or not candidate['resume_key']:
            return jsonify(error='No résumé uploaded for this candidate.'), 404
        try:
            if candidate['resume_storage'] == 's3':
                url = s3_client().generate_presigned_url('get_object', Params={
                    'Bucket':app.config['S3_BUCKET'], 'Key':candidate['resume_key'],
                    'ResponseContentDisposition':f'attachment; filename="{candidate["resume_name"]}"'}, ExpiresIn=300)
                return redirect(url)
            return send_file(current_app.config['DATA_DIR'] / candidate['resume_key'], as_attachment=True, download_name=candidate['resume_name'])
        except Exception:
            app.logger.exception('Resume download failed')
            return jsonify(error='The document is unavailable. Check storage and try again.'), 503

    @app.get('/api/projects')
    def projects():
        return jsonify(projects=[dict(row) for row in database().execute('SELECT * FROM projects ORDER BY id DESC')])

    @app.post('/api/projects')
    @permitted('manager')
    def add_project():
        payload = request.get_json(silent=True)
        try:
            if not isinstance(payload, dict):
                raise ValueError('Provide project details.')
            name, skills = text_field(payload, 'name'), skill_list(payload)
        except ValueError as error:
            return jsonify(error=str(error)), 400
        cursor = database().execute('INSERT INTO projects (name,skills) VALUES (?,?)', (name,skills))
        database().commit()
        return jsonify(id=cursor.lastrowid), 201

    @app.get('/api/projects/<int:project_id>/matches')
    def project_matches(project_id):
        project = database().execute('SELECT * FROM projects WHERE id=?', (project_id,)).fetchone()
        if not project:
            return jsonify(error='Project not found.'), 404
        required = set(project['skills'].split(', '))
        shortlisted = {row[0] for row in database().execute('SELECT candidate_id FROM shortlists WHERE project_id=?', (project_id,))}
        matches = []
        for row in database().execute('SELECT id,name,skills,status FROM candidates'):
            matched = sorted(required & set(row['skills'].split(', ')))
            matches.append(dict(row) | {'match_percent':round(100 * len(matched) / len(required)),
                'matched_skills':matched, 'missing_skills':sorted(required - set(matched)),
                'shortlisted':row['id'] in shortlisted})
        matches.sort(key=lambda c: (-c['match_percent'], c['name'].lower(), c['id']))
        return jsonify(project=dict(project), candidates=matches)

    @app.post('/api/projects/<int:project_id>/shortlist')
    @permitted('manager')
    def shortlist(project_id):
        payload = request.get_json(silent=True)
        candidate_id = payload.get('candidate_id') if isinstance(payload, dict) else None
        if type(candidate_id) is not int:
            return jsonify(error='Choose a candidate.'), 400
        if not database().execute('SELECT id FROM projects WHERE id=?', (project_id,)).fetchone() or not database().execute('SELECT id FROM candidates WHERE id=?', (candidate_id,)).fetchone():
            return jsonify(error='Project or candidate not found.'), 404
        database().execute('INSERT OR IGNORE INTO shortlists VALUES (?,?)', (project_id,candidate_id))
        database().commit()
        return jsonify(ok=True)

    @app.delete('/api/projects/<int:project_id>/shortlist/<int:candidate_id>')
    @permitted('manager')
    def remove_shortlist(project_id, candidate_id):
        database().execute('DELETE FROM shortlists WHERE project_id=? AND candidate_id=?', (project_id,candidate_id))
        database().commit()
        return jsonify(ok=True)
