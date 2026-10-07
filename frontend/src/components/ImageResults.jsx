import TaskCard from './TaskCard';
import ExplanationBlock from './ExplanationBlock';
import CommonsensePanel from './CommonsensePanel';
import QAPanel from './QAPanel';
import VisualConceptsPanel from './VisualConceptsPanel';
import TimingsFooter from './TimingsFooter';
import { formatTime } from '../lib/format';

/** Full results for one analysed image (or one moment of a video). */
export default function ImageResults({ result, analyzedAt, title = 'Analysis results', showFooter = true }) {
  const { prediction, visual, commonsense, frame, timings_s: timings } = result;
  const quality = frame?.quality;
  const people = prediction.people_count;

  return (
    <div className="results-dashboard">
      <div className="card summary-bar">
        <div className="summary-left">
          <span className="summary-title">{title}</span>
          <span className="tag">
            {people} {people === 1 ? 'person' : 'people'} detected
          </span>
          {quality?.is_blurry && <span className="tag tag-warn">Image is not sharp; results may be less reliable</span>}
          {quality?.enhancements?.length > 0 && <span className="tag">Low-light correction applied</span>}
        </div>
        {analyzedAt && <span className="muted num">{formatTime(analyzedAt)}</span>}
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
      {showFooter && <TimingsFooter timings={timings} result={result} />}
    </div>
  );
}
