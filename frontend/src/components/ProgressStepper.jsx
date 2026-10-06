import { STAGE_ORDER, STAGE_META } from '../lib/api';
import { CheckIcon } from './Icons';

/**
 * Mirrors the four Phase I modules. `doneStages` and `activeStage` come
 * straight from the backend's job-status poll, so this reflects what the
 * pipeline is actually doing, not a simulated progress bar.
 */
export default function ProgressStepper({ doneStages = [], activeStage, compact = false }) {
  return (
    <div className="stepper">
      {STAGE_ORDER.map((stage) => {
        const isDone = doneStages.includes(stage);
        const isActive = !isDone && stage === activeStage;
        const cls = isDone ? 'done' : isActive ? 'active' : 'pending';
        return (
          <div className={`step ${cls}`} key={stage}>
            <span className="step-rail" />
            <span className="step-dot">
              {isDone && <CheckIcon />}
              {isActive && <span className="ring" />}
            </span>
            <span className="step-label">{compact ? STAGE_META[stage].short : STAGE_META[stage].full}</span>
          </div>
        );
      })}
    </div>
  );
}
