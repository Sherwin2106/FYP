import { useEffect, useState } from 'react';
import { ImageIcon, AlertIcon } from './Icons';
import ProgressStepper from './ProgressStepper';

export function EmptyState() {
  return (
    <div className="card state-panel">
      <ImageIcon className="state-icon" />
      <p className="state-title">No analysis yet</p>
      <p className="state-sub">
        Upload an image or capture one with the camera, then select <strong>Analyze</strong>. The results will show the
        predicted activity, relationship and intention, each with a confidence score, an explanation and the supporting
        commonsense knowledge.
      </p>
    </div>
  );
}

export function ErrorState({ message, onRetry }) {
  return (
    <div className="card state-panel error-panel">
      <AlertIcon className="state-icon" />
      <p className="state-title">The analysis could not be completed</p>
      {message && <p className="error-detail">{message}</p>}
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
    const id = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="card state-panel">
      <p className="state-title">Analyzing the image</p>
      <p className="muted num">Elapsed time: {elapsed} s</p>
      <div className="working-stepper">
        <ProgressStepper doneStages={doneStages} activeStage={activeStage} />
      </div>
      <p className="state-sub">This usually takes under a minute.</p>
    </div>
  );
}
