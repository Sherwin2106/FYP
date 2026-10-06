// Thin wrapper around the FastAPI backend (backend/main.py). In dev, Vite
// proxies /api/* to http://127.0.0.1:8000 (see vite.config.js); in production
// the built frontend is served by the same FastAPI process, so a relative
// path works in both cases.

async function readError(response) {
  try {
    const body = await response.json();
    return body.detail || body.message || null;
  } catch {
    return null;
  }
}

export async function fetchHealth() {
  const res = await fetch('/api/health');
  if (!res.ok) {
    throw new Error((await readError(res)) || `Health check failed (${res.status})`);
  }
  return res.json();
}

/**
 * Upload an image and start Phase I analysis.
 * @returns {Promise<{job_id: string}>}
 */
export async function startAnalysis({ file, question, choices }) {
  const form = new FormData();
  form.append('image', file, file.name || `capture-${Date.now()}.jpg`);
  if (question && question.trim()) form.append('question', question.trim());
  const cleanChoices = (choices || []).map((c) => c.trim()).filter(Boolean);
  if (cleanChoices.length >= 2) form.append('choices', cleanChoices.join('|'));

  const res = await fetch('/api/analyze', { method: 'POST', body: form });
  if (!res.ok) {
    throw new Error((await readError(res)) || `Upload failed (${res.status})`);
  }
  return res.json();
}

export async function pollJob(jobId, { signal } = {}) {
  const res = await fetch(`/api/jobs/${jobId}`, { signal });
  if (!res.ok) {
    throw new Error((await readError(res)) || `Could not reach job ${jobId} (${res.status})`);
  }
  return res.json();
}

export const STAGE_ORDER = ['preprocess', 'vlm', 'conceptnet', 'reasoning'];

export const STAGE_META = {
  preprocess: { short: 'Preprocessing', full: 'Module 1 — Input & preprocessing (OpenCV)' },
  vlm: { short: 'SmolVLM reasoning', full: 'Module 2 — Vision-language understanding & draft reasoning' },
  conceptnet: { short: 'ConceptNet lookup', full: 'Module 3 — Commonsense knowledge enrichment' },
  reasoning: { short: 'Fusing prediction', full: 'Module 4 — Social interaction reasoning' },
};
