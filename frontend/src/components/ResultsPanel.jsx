import { useEffect, useState } from 'react';
import { EmptyState, ErrorState, WorkingState } from './StatePanels';
import ImageResults from './ImageResults';
import VideoResults, { MomentStrip } from './VideoResults';
import ProgressStepper from './ProgressStepper';

function VideoWorkingState({ progress, doneStages, activeStage }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, []);
  const moments = progress?.moments || [];
  const done = moments.filter((m) => m.status === 'done' || m.status === 'error').length;
  const scanning = moments.length === 0;

  return (
    <div className="card card-body video-working">
      <p className="state-title">Analyzing the video</p>
      <p className="muted num">Elapsed time: {elapsed} s</p>
      {scanning ? (
        <div className="scan">
          <p>Selecting key moments…</p>
          <div className="confidence-track">
            <div className="confidence-fill" style={{ width: `${Math.round((progress?.scan || 0) * 100)}%` }} />
          </div>
        </div>
      ) : (
        <>
          <p className="video-working-count num">
            Moment {Math.min(done + 1, moments.length)} of {moments.length}
          </p>
          <MomentStrip moments={moments} current={progress.current} />
          <div className="working-stepper" style={{ alignSelf: 'center' }}>
            <ProgressStepper doneStages={doneStages} activeStage={activeStage} />
          </div>
        </>
      )}
      <p className="state-sub" style={{ alignSelf: 'center' }}>
        Each moment takes about half a minute. Results appear when all moments are done.
      </p>
    </div>
  );
}

export default function ResultsPanel({
  status,
  activeStage,
  doneStages,
  error,
  result,
  onRetry,
  analyzedAt,
  isVideo,
  videoProgress,
}) {
  if (status === 'error') {
    return <ErrorState message={error} onRetry={onRetry} />;
  }
  if (status === 'queued' || status === 'running') {
    return isVideo ? (
      <VideoWorkingState progress={videoProgress} doneStages={doneStages} activeStage={activeStage} />
    ) : (
      <WorkingState doneStages={doneStages} activeStage={activeStage} />
    );
  }
  if (status !== 'done' || !result) {
    return <EmptyState />;
  }
  if (result.kind === 'video') {
    return <VideoResults key={analyzedAt} result={result} analyzedAt={analyzedAt} />;
  }
  return <ImageResults result={result} analyzedAt={analyzedAt} />;
}
