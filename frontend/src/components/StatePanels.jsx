import { useEffect, useState } from 'react';
import { SparkIcon, AlertIcon } from './Icons';
import ProgressStepper from './ProgressStepper';

export function EmptyState() {
  return (
    <div className="card state-panel">
      <SparkIcon className="state-icon" />
      <p className="state-title">Nothing analyzed yet</p>
      <p className="state-sub">
        Upload a photo (or capture one from your camera) on the left, then press <em>Analyze interaction</em> to see
        the predicted activity, relationship and intention — each with its confidence, the SmolVLM caption behind it,
        and the ConceptNet evidence that shaped the final answer.
      </p>
    </div>
  );
}

export function ErrorState({ message, onRetry }) {
  return (
    <div className="card state-panel error-panel">
      <AlertIcon className="state-icon" />
      <p className="state-title">Something went wrong</p>
      {message && <p className="error-detail mono">{message}</p>}
      {onRetry && (
        <button type="button" className="btn btn-primary" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}

export function WorkingState({ doneStages, activeStage }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    setElapsed(0);
    const id = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="card state-panel">
      <SparkIcon className="state-icon" />
      <p className="state-title">Reading the scene…</p>
      <p className="working-elapsed">{elapsed}s elapsed</p>
      <div className="working-stepper">
        <ProgressStepper doneStages={doneStages} activeStage={activeStage} />
      </div>
      <p className="state-sub">
        SmolVLM is doing the heavy lifting here — expect 15–45s depending on your hardware and the image size.
      </p>
    </div>
  );
}
