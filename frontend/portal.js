'use strict';
const $ = id => document.getElementById(id);
let role, csrf, candidates = [], selectedId, schema;
function el(tag, text, className) { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; if (className) node.className = className; return node; }
function notice(message, error = false) { $('portal-error').hidden = true; $('portal-message').hidden = true; const target = $(error ? 'portal-error' : 'portal-message'); target.textContent = message; target.hidden = false; }
async function api(path, method = 'GET', payload) {
  const options = {method, headers: {'X-CSRF-Token': csrf || ''}};
  if (payload instanceof FormData) options.body = payload;
  else if (payload !== undefined) { options.headers['Content-Type'] = 'application/json'; options.body = JSON.stringify(payload); }
  const response = await fetch(path, options);
  const result = await response.json();
  if (!response.ok) { if (response.status === 401 && role) { role = null; showLogin(); } throw new Error(result.error || 'Request failed. Try again.'); }
  return result;
}
async function action(button, work) { button.disabled = true; $('portal-error').hidden = true; $('portal-message').hidden = true; try { await work(); } catch (error) { notice(error.message, true); } finally { button.disabled = false; } }
function showLogin() { $('login-view').hidden = false; $('workspace-view').hidden = true; $('account').hidden = true; $('candidate-detail').hidden = true; selectedId = null; candidates = []; $('candidate-list').replaceChildren(); $('matches').replaceChildren(); $('login-form').reset(); }
async function loadWorkspace() {
  $('login-view').hidden = true; $('workspace-view').hidden = false; $('account').hidden = false;
  $('role-label').textContent = role === 'hr' ? 'HR' : 'Project manager';
  $('workspace-title').textContent = role === 'hr' ? 'Recruitment workspace' : 'Build your project team';
  $('workspace-description').textContent = role === 'hr' ? 'Keep candidate records, assessments and hiring stages together.' : 'Compare required skills and save a shortlist for HR.';
  $('candidate-form').hidden = role !== 'hr'; $('project-form').hidden = role !== 'manager'; $('assessment-link').hidden = role !== 'hr';
  const session = await api('/api/session'); csrf = session.csrf; $('storage-label').textContent = `Document storage: ${session.storage}`;
  schema = await api('/api/schema');
  $('classifier-summary').hidden = role !== 'hr';
  $('classifier-summary').textContent = `${schema.model} classifier · predicts Selected / Not Selected from completed assessments · trained on synthetic data.`;
  await refreshCandidates(); await refreshProjects(); switchView(role === 'hr' ? 'candidates' : 'projects');
}
function switchView(view) { $('candidates-view').hidden = view !== 'candidates'; $('projects-view').hidden = view !== 'projects'; for (const mode of ['candidates', 'projects']) { $(`${mode}-tab`).classList.toggle('active', mode === view); $(`${mode}-tab`).setAttribute('aria-pressed', String(mode === view)); } }
function makeTable(headers) { const table = el('table'); const head = el('thead'), row = el('tr'); for (const text of headers) { const th = el('th', text); th.scope = 'col'; row.append(th); } head.append(row); const body = el('tbody'); table.append(head, body); return [table, body]; }
function stage(text) { return el('span', text, `stage ${text === 'Selected' ? 'stage-selected' : ''}`); }
async function refreshCandidates() { candidates = (await api('/api/candidates')).candidates; renderCandidates(); }
function renderCandidates() {
  const query = $('search').value.trim().toLowerCase(), status = $('status-filter').value;
  const visible = candidates.filter(c => `${c.name} ${c.skills}`.toLowerCase().includes(query) && (!status || c.status === status));
  if (!visible.length) { $('candidate-list').replaceChildren(el('p', candidates.length ? 'No candidates match your filters.' : 'No candidates yet. HR can add the first candidate above.', 'empty-state')); return; }
  const headers = role === 'hr' ? ['Candidate', 'Skills', 'Stage', 'Predicted selection', 'Action'] : ['Candidate', 'Skills', 'Stage'];
  const [table, body] = makeTable(headers);
  for (const c of visible) { const row = el('tr'), name = el('td'); name.append(el('span', c.name, 'candidate-name')); if (role === 'hr') name.append(el('span', c.email, 'candidate-email')); const statusCell = el('td'); statusCell.append(stage(c.status)); row.append(name, el('td', c.skills), statusCell);
    if (role === 'hr') { const result = el('td', c.predicted_selection || 'Not assessed'); if (c.model_score !== null) result.append(el('span', `Score ${c.model_score.toFixed(3)}`, 'match-reason')); const cell = el('td'), button = el('button', 'View / assess', 'secondary'); button.type = 'button'; button.setAttribute('aria-label', `View or assess ${c.name}`); button.addEventListener('click', () => openCandidate(c.id)); cell.append(button); row.append(result, cell); } body.append(row); }
  $('candidate-list').replaceChildren(table);
}
function openCandidate(id) {
  const c = candidates.find(row => row.id === id); selectedId = id;
  $('candidate-detail').hidden = false; $('detail-name').textContent = c.name; $('detail-summary').textContent = `${c.email} · ${c.skills}`;
  $('status-form').elements.status.value = c.status; $('resume-form').reset();
  $('resume-download').hidden = !c.has_resume; $('resume-download').href = `/api/candidates/${id}/resume`; $('resume-download').textContent = `Download ${c.resume_name || 'résumé'}`;
  $('saved-fields').replaceChildren();
  for (const f of schema.features) { const label = el('label', f.label), input = el('input'); Object.assign(input, {type:'number', name:f.key, min:f.min, max:f.max, step:f.key === 'cgpa' ? '0.1' : '1', required:true}); input.value = c.scores?.[f.key] ?? ''; label.append(input, el('small', `${f.min}–${f.max}`)); $('saved-fields').append(label); }
  $('saved-result').replaceChildren(el('p', c.predicted_selection ? `Predicted selection: ${c.predicted_selection} · model score ${c.model_score.toFixed(3)}.` : 'No assessment saved yet.'));
  $('candidate-detail').scrollIntoView({block:'start'}); $('close-detail').focus({preventScroll:true});
}
async function refreshProjects() {
  const previous = $('project-select').value; const data = await api('/api/projects'); $('project-select').replaceChildren(new Option(data.projects.length ? 'Choose a project' : 'No projects yet', ''));
  for (const project of data.projects) $('project-select').append(new Option(project.name, project.id));
  $('project-select').value = previous; if ($('project-select').value) await showMatches(); else $('matches').replaceChildren(el('p', 'Choose a project to compare candidates and see its shortlist.', 'empty-state'));
}
async function showMatches() {
  const projectId = $('project-select').value; if (!projectId) { $('matches').replaceChildren(); return; }
  const data = await api(`/api/projects/${projectId}/matches`); $('match-description').textContent = `Required skills: ${data.project.skills}. Match = listed required skills ÷ total required skills. Shortlists are visible to HR.`;
  if (!data.candidates.length) { $('matches').replaceChildren(el('p', 'HR needs to add candidates before you can make a shortlist.', 'empty-state')); return; }
  const [table, body] = makeTable(['Candidate', 'Skill match', 'Stage', 'Shortlist']);
  for (const c of data.candidates) { const row = el('tr'), match = el('td'), status = el('td'), choice = el('td'); match.append(el('span', `${c.match_percent}%`, 'match-score'), el('span', `Matched: ${c.matched_skills.join(', ') || 'none'}`, 'match-reason')); if (c.missing_skills.length) match.append(el('span', `Missing: ${c.missing_skills.join(', ')}`, 'match-reason')); status.append(stage(c.status));
    if (role === 'manager') { const button = el('button', c.shortlisted ? 'Remove from shortlist' : 'Shortlist', 'secondary'); button.setAttribute('aria-label', `${c.shortlisted ? 'Remove' : 'Shortlist'} ${c.name}`); button.addEventListener('click', () => action(button, async () => { if (c.shortlisted) await api(`/api/projects/${projectId}/shortlist/${c.id}`, 'DELETE'); else await api(`/api/projects/${projectId}/shortlist`, 'POST', {candidate_id:c.id}); await showMatches(); notice(c.shortlisted ? 'Candidate removed from shortlist.' : 'Candidate shortlisted. HR can see this project shortlist.'); })); choice.append(button); }
    else choice.append(el('span', c.shortlisted ? 'Shortlisted by manager' : 'Not shortlisted'));
    row.append(el('td', c.name), match, status, choice); body.append(row); }
  $('matches').replaceChildren(table);
}
$('login-form').addEventListener('submit', event => { event.preventDefault(); action(event.submitter, async () => { const result = await api('/api/login', 'POST', Object.fromEntries(new FormData(event.target))); role = result.role; csrf = result.csrf; event.target.reset(); await loadWorkspace(); }); });
$('logout').addEventListener('click', () => action($('logout'), async () => { await api('/api/logout', 'POST', {}); role = null; showLogin(); csrf = (await api('/api/session')).csrf; }));
$('candidate-form').addEventListener('submit', event => { event.preventDefault(); action(event.submitter, async () => { const result = await api('/api/candidates', 'POST', Object.fromEntries(new FormData(event.target))); event.target.reset(); await refreshCandidates(); openCandidate(result.id); notice('Candidate saved. Add their résumé or interview assessment below.'); }); });
$('status-form').addEventListener('submit', event => { event.preventDefault(); action(event.submitter, async () => { await api(`/api/candidates/${selectedId}/status`, 'PATCH', {status:event.target.elements.status.value}); await refreshCandidates(); notice('Hiring stage saved.'); }); });
$('resume-form').addEventListener('submit', event => { event.preventDefault(); action(event.submitter, async () => { const file = event.target.elements.file.files[0]; if (!file || file.size > 5 * 1024 * 1024) throw new Error('Choose a PDF smaller than 5 MB.'); await api(`/api/candidates/${selectedId}/resume`, 'POST', new FormData(event.target)); await refreshCandidates(); openCandidate(selectedId); notice('Résumé uploaded.'); }); });
$('saved-assessment-form').addEventListener('submit', event => { event.preventDefault(); action(event.submitter, async () => { const scores = Object.fromEntries([...new FormData(event.target)].map(([key,value]) => [key, Number(value)])); const result = await api(`/api/candidates/${selectedId}/assessment`, 'POST', {scores}); await refreshCandidates(); $('saved-result').replaceChildren(el('p', `Predicted selection: ${result.predicted_selection} · ${schema.model}. Model score ${result.model_score.toFixed(3)} · threshold ${result.threshold.toFixed(2)}.`)); notice('Selection prediction and assessment saved.'); }); });
$('project-form').addEventListener('submit', event => { event.preventDefault(); action(event.submitter, async () => { const result = await api('/api/projects', 'POST', Object.fromEntries(new FormData(event.target))); event.target.reset(); await refreshProjects(); $('project-select').value = result.id; await showMatches(); notice('Project saved. Compare candidates below.'); }); });
$('project-select').addEventListener('change', () => showMatches().catch(error => notice(error.message, true)));
$('search').addEventListener('input', renderCandidates); $('status-filter').addEventListener('change', renderCandidates);
$('close-detail').addEventListener('click', () => { $('candidate-detail').hidden = true; selectedId = null; });
for (const view of ['candidates', 'projects']) $(`${view}-tab`).addEventListener('click', () => { switchView(view); if (view === 'projects') refreshProjects().catch(error => notice(error.message, true)); });
(async () => { try { const data = await api('/api/session'); role = data.role; csrf = data.csrf; if (role) await loadWorkspace(); } catch (error) { notice('Could not connect to the server. Refresh and try again.', true); } })();
