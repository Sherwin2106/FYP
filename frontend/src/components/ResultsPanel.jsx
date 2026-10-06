import { EmptyState, ErrorState, WorkingState } from './StatePanels';
import TaskCard from './TaskCard';
import ExplanationBlock from './ExplanationBlock';
import CommonsensePanel from './CommonsensePanel';
import QAPanel from './QAPanel';
import VisualConceptsPanel from './VisualConceptsPanel';
import TimingsFooter from './TimingsFooter';
import RawJsonViewer from './RawJsonViewer';
import { PeopleIcon } from './Icons';
import { formatTime } from '../lib/format';

export default function ResultsPanel({ status, activeStage, doneStages, error, result, onRetry, analyzedAt }) {
  if (status === 'error') {
    return <ErrorState message={error} onRetry={onRetry} />;
  }
  if (status === 'queued' || status === 'running') {
    return <WorkingState doneStages={doneStages} activeStage={activeStage} />;
  }
  if (status !== 'done' || !result) {
    return <EmptyState />;
  }

  const { prediction, visual, commonsense, frame, timings_s: timings, model_id: modelId } = result;
  const quality = frame?.quality;
  const enhanced = quality?.enhancements?.length > 0;

  return (
    <div className="results-dashboard">
      <div className="card results-summary-bar">
        <div className="results-summary-left">
          <span className="chip">
            <PeopleIcon style={{ width: 13, height: 13 }} />
            {prediction.people_count} {prediction.people_count === 1 ? 'person' : 'people'}
          </span>
          <span className="muted mono" style={{ fontSize: 11.5 }}>
            analyzed {formatTime(analyzedAt)}
          </span>
          {quality?.is_blurry && (
            <span className="chip" style={{ background: 'var(--warning)', color: 'var(--card)' }}>
              blurry input — predictions may be less reliable
            </span>
          )}
          {enhanced && (
            <span className="chip" title={quality.enhancements.join(', ')}>
              low-light image was auto-enhanced
            </span>
          )}
        </div>
      </div>

      <div className="task-grid">
        <TaskCard title="Activity" task={prediction.activity} />
        <TaskCard title="Relationship" task={prediction.relationship} />
        <TaskCard title="Intention" task={prediction.intention} />
      </div>

      <ExplanationBlock explanation={prediction.explanation} />
      <QAPanel prediction={prediction} />
      <CommonsensePanel evidence={prediction.evidence} commonsense={commonsense} />
      <VisualConceptsPanel visual={visual} />
      <TimingsFooter timings={timings} modelId={modelId} />
      <RawJsonViewer data={result} />
    </div>
  );
}
