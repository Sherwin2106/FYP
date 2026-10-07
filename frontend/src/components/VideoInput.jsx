import { useState } from 'react';
import { VideoIcon } from './Icons';

const MAX_BYTES = 300 * 1024 * 1024;

export default function VideoInput({ videoUrl, onFile, disabled }) {
  const [error, setError] = useState(null);

  function handle(fileList) {
    const file = fileList?.[0];
    if (!file) return;
    if (!file.type.startsWith('video/') && !/\.(mp4|mov|avi|webm|mkv|m4v)$/i.test(file.name || '')) {
      setError('Please choose an MP4, MOV, AVI or WEBM video.');
      return;
    }
    if (file.size > MAX_BYTES) {
      setError('The video is larger than 300 MB. Please choose a shorter clip.');
      return;
    }
    setError(null);
    onFile(file);
  }

  return (
    <div>
      {videoUrl ? (
        <div className="video-preview">
          <video src={videoUrl} controls muted playsInline />
          <label className={`btn btn-secondary btn-sm video-change${disabled ? ' disabled' : ''}`}>
            Change video
            <input type="file" accept="video/*" disabled={disabled} onChange={(e) => handle(e.target.files)} hidden />
          </label>
        </div>
      ) : (
        <label className="dropzone">
          <div className="dropzone-placeholder">
            <VideoIcon />
            <p className="dropzone-placeholder-title">Click to choose a video</p>
            <p className="dropzone-placeholder-sub">MP4, MOV or WEBM, up to 300 MB</p>
          </div>
          <input type="file" accept="video/*" disabled={disabled} onChange={(e) => handle(e.target.files)} />
        </label>
      )}
      {error && <p className="field-error">{error}</p>}
      <p className="video-note">
        Up to 7 key moments are selected from the video and analyzed one after another (about half a minute each).
      </p>
    </div>
  );
}
