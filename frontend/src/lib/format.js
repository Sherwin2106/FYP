export function pct(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return `${Math.round(value * 100)}%`;
}

export function confidenceTier(task) {
  if (!task) return 'low';
  if (task.ambiguous) return 'warn';
  if (task.confidence >= 0.6) return 'high';
  if (task.confidence >= 0.35) return 'warn';
  return 'low';
}

export function shortModelName(modelId) {
  if (!modelId) return 'model';
  const last = modelId.split('/').pop();
  return last.replace('HuggingFaceTB-', '');
}

const TIMING_LABELS = {
  module1_preprocess: 'preprocess',
  module2_vlm: 'smolvlm',
  module3_conceptnet: 'conceptnet',
  module4_reasoning: 'reasoning',
  total: 'total',
};

export function timingLabel(key) {
  return TIMING_LABELS[key] || key;
}

export function formatSeconds(value) {
  if (value === null || value === undefined) return '—';
  return `${value.toFixed(value < 10 ? 2 : 1)}s`;
}

export function titleCase(text) {
  if (!text) return '';
  return text.replace(/\b\w/g, (c) => c.toUpperCase());
}

export function formatTime(ts) {
  try {
    return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch {
    return '';
  }
}

export function cleanEvidence(text) {
  return (text || '').replace(/^ConceptNet:\s*/i, '');
}
