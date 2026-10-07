import { useState } from 'react';
import ImageResults from './ImageResults';
import TimingsFooter from './TimingsFooter';
import { formatTime } from '../lib/format';

export function formatClock(seconds) {
  const s = Math.max(0, Math.round(seconds || 0));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

/** Thumbnails of the selected moments, in time order; used both while working and for results. */
export function MomentStrip({ moments, selected, onSelect, current }) {
  return (
    <div className="moment-strip">
      {moments.map((m) => {
        const status = m.status || (m.result ? 'done' : m.error ? 'error' : 'pending');
        const clickable = onSelect && status === 'done';
        return (
          <button
            type="button"
            key={m.index}
            className={`moment${selected === m.index ? ' active' : ''}${current === m.index ? ' current' : ''} moment-${status}`}
            onClick={clickable ? () => onSelect(m.index) : undefined}
            disabled={!clickable}
          >
            <img src={m.thumbnail} alt={`Moment at ${formatClock(m.time_s)}`} />
            <span className="moment-time num">{formatClock(m.time_s)}</span>
            <span className="moment-label">
              {status === 'done' && m.summary ? m.summary.activity : null}
              {status === 'running' ? 'Analyzing…' : null}
              {status === 'pending' ? 'Waiting' : null}
              {status === 'error' ? 'Could not analyze' : null}
            </span>
          </button>
        );
      })}
    </div>
  );
}

export default function VideoResults({ result, analyzedAt }) {
  const moments = result.moments || [];
  const firstDone = moments.find((m) => m.result);
  const [selected, setSelected] = useState(firstDone ? firstDone.index : null);
  const chosen = moments.find((m) => m.index === selected && m.result);
  const summary = result.summary || {};
  const n = moments.filter((m) => m.result).length;
  const info = result.video || {};
  const totalTime = moments.reduce((sum, m) => {
    const t = m.result?.timings_s;
    return t ? sum + Object.values(t).reduce((a, b) => a + b, 0) : sum;
  }, 0);

  return (
    <div className="results-dashboard">
      <div className="card summary-bar">
        <div className="summary-left">
          <span className="summary-title">Video analysis</span>
          <span className="tag num">Duration {formatClock(info.duration_s)}</span>
          <span className="tag num">
            {n} key {n === 1 ? 'moment' : 'moments'} analyzed
          </span>
        </div>
        {analyzedAt && <span className="muted num">{formatTime(analyzedAt)}</span>}
      </div>

      <section className="card card-body">
        <h2 className="section-title">Overall interpretation</h2>
        <div className="video-summary">
          {[
            ['Activity', summary.activity],
            ['Relationship', summary.relationship],
            ['Intention', summary.intention],
          ].map(([title, s]) =>
            s ? (
              <div key={title}>
                <p className="label">{title}</p>
                <p className="video-summary-label">{s.label}</p>
                <p className="muted num" style={{ fontSize: 13 }}>
                  In {s.moments} of {n} moments · average confidence {Math.round(s.mean_confidence * 100)}%
                </p>
              </div>
            ) : null,
          )}
        </div>
      </section>

      <section className="card card-body">
        <h2 className="section-title">Key moments</h2>
        <p className="muted" style={{ fontSize: 13.5, marginBottom: 12 }}>
          Selected as the sharpest frame of each scene, spread across the whole video. Select a moment to see its full
          analysis.
        </p>
        <MomentStrip moments={moments} selected={selected} onSelect={setSelected} />
      </section>

      {chosen && (
        <>
          <div className="moment-detail-head card">
            <img src={chosen.thumbnail} alt="" />
            <div>
              <p className="label" style={{ marginBottom: 2 }}>
                Moment at {formatClock(chosen.time_s)}
              </p>
              <p className="muted" style={{ fontSize: 13 }}>
                Represents the scene from {formatClock(chosen.segment?.[0])} to {formatClock(chosen.segment?.[1])}
              </p>
            </div>
          </div>
          <ImageResults result={chosen.result} title="Moment results" showFooter={false} />
        </>
      )}

      <TimingsFooter timings={{ total: totalTime }} result={result} />
    </div>
  );
}
