import Dropzone from './Dropzone';
import CameraCapture from './CameraCapture';
import QuestionForm from './QuestionForm';
import ProgressStepper from './ProgressStepper';
import { ImageIcon, CameraIcon } from './Icons';

export default function SourcePanel({
  mode,
  setMode,
  previewUrl,
  onFile,
  question,
  setQuestion,
  choices,
  setChoices,
  onAnalyze,
  status,
  activeStage,
  doneStages,
  hasFile,
}) {
  const busy = status === 'queued' || status === 'running';

  return (
    <div className="card source-panel">
      <div className="tabs" role="tablist" aria-label="Image source">
        <button
          type="button"
          role="tab"
          aria-selected={mode === 'upload'}
          className={`tab${mode === 'upload' ? ' active' : ''}`}
          onClick={() => setMode('upload')}
          disabled={busy}
        >
          <ImageIcon /> Upload
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={mode === 'camera'}
          className={`tab${mode === 'camera' ? ' active' : ''}`}
          onClick={() => setMode('camera')}
          disabled={busy}
        >
          <CameraIcon /> Camera
        </button>
      </div>

      {mode === 'upload' ? (
        <Dropzone previewUrl={previewUrl} onFile={onFile} disabled={busy} />
      ) : (
        <CameraCapture onFile={onFile} disabled={busy} />
      )}

      <QuestionForm
        question={question}
        setQuestion={setQuestion}
        choices={choices}
        setChoices={setChoices}
        disabled={busy}
      />

      <div className="analyze-bar">
        {busy ? (
          <ProgressStepper doneStages={doneStages} activeStage={activeStage} compact />
        ) : (
          <button type="button" className="btn btn-primary analyze-btn" onClick={onAnalyze} disabled={!hasFile}>
            Analyze interaction →
          </button>
        )}
        {!busy && <p className="analyze-hint">Runs SmolVLM + ConceptNet locally — usually 15–45s.</p>}
      </div>
    </div>
  );
}
