import { useRef, useState } from 'react';
import { ImageIcon } from './Icons';

const MAX_BYTES = 15 * 1024 * 1024;
const ACCEPTED = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];

export default function Dropzone({ previewUrl, onFile, disabled }) {
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  function validate(file) {
    if (!file) return 'No file selected.';
    if (ACCEPTED.length && !ACCEPTED.includes(file.type) && !/\.(jpe?g|png|webp|bmp)$/i.test(file.name || '')) {
      return 'Please choose a JPG, PNG, WEBP or BMP image.';
    }
    if (file.size > MAX_BYTES) return 'The image is larger than 15 MB. Please choose a smaller file.';
    return null;
  }

  function handleFiles(fileList) {
    const file = fileList?.[0];
    const problem = validate(file);
    if (problem) {
      setError(problem);
      return;
    }
    setError(null);
    onFile(file);
  }

  return (
    <div>
      <div
        className={`dropzone${dragging ? ' dragging' : ''}`}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (!disabled) handleFiles(e.dataTransfer.files);
        }}
      >
        {previewUrl ? (
          <>
            <img className="dropzone-preview" src={previewUrl} alt="Selected input" />
            <span className="dropzone-change">Change image</span>
          </>
        ) : (
          <div className="dropzone-placeholder">
            <ImageIcon />
            <p className="dropzone-placeholder-title">Drag an image here or click to browse</p>
            <p className="dropzone-placeholder-sub">JPG, PNG or WEBP, up to 15 MB</p>
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          disabled={disabled}
          onChange={(e) => handleFiles(e.target.files)}
          aria-label="Upload an image"
        />
      </div>
      {error && <p className="field-error">{error}</p>}
    </div>
  );
}
