import Dropzone from './Dropzone';
import CameraCapture from './CameraCapture';
import VideoInput from './VideoInput';
import QuestionForm from './QuestionForm';
import { ImageIcon, CameraIcon, VideoIcon } from './Icons';

const TABS = [
  ['upload', 'Upload image', ImageIcon],
  ['camera', 'Camera', CameraIcon],
  ['video', 'Video', VideoIcon],
];

export default function SourcePanel({
  mode,
  setMode,
  previewUrl,
  onFile,
  videoUrl,
  onVideoFile,
  question,
  setQuestion,
  choices,
  setChoices,
  onAnalyze,
  status,
  hasFile,
}) {
  const busy = status === 'queued' || status === 'running';

  return (
    <div className="card card-body">
      <h2 className="section-title">Input</h2>
      <div className="tabs" role="tablist" aria-label="Input source">
        {TABS.map(([key, label, Icon]) => (
          <button
            key={key}
            type="button"
            role="tab"
            aria-selected={mode === key}
            className={`tab${mode === key ? ' active' : ''}`}
            onClick={() => setMode(key)}
            disabled={busy}
          >
            <Icon /> {label}
          </button>
        ))}
      </div>

      {mode === 'upload' && <Dropzone previewUrl={previewUrl} onFile={onFile} disabled={busy} />}
      {mode === 'camera' && <CameraCapture onFile={onFile} disabled={busy} />}
      {mode === 'video' && <VideoInput videoUrl={videoUrl} onFile={onVideoFile} disabled={busy} />}

      <QuestionForm
        question={question}
        setQuestion={setQuestion}
        choices={choices}
        setChoices={setChoices}
        disabled={busy}
        allowChoices={mode !== 'video'}
      />

      <div className="analyze-bar">
        <button type="button" className="btn btn-primary analyze-btn" onClick={onAnalyze} disabled={!hasFile || busy}>
          {busy ? 'Analyzing…' : mode === 'video' ? 'Analyze video' : 'Analyze'}
        </button>
      </div>
    </div>
  );
}
