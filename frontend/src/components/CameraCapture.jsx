import { useEffect, useRef, useState } from 'react';
import { CameraIcon } from './Icons';

/**
 * Live camera tab. Mounted only while the "Camera" source tab is active, so the
 * stream is requested on mount and released (all tracks stopped) on unmount —
 * the camera light should never stay on after the user switches away.
 */
export default function CameraCapture({ onFile, disabled }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const [error, setError] = useState(null);
  const [ready, setReady] = useState(false);
  const [capturedUrl, setCapturedUrl] = useState(null);

  useEffect(() => {
    let cancelled = false;
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('This browser does not support camera access.');
      return undefined;
    }
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 960 } } })
      .then((stream) => {
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setReady(true);
      })
      .catch((err) => {
        if (cancelled) return;
        const message =
          err.name === 'NotAllowedError'
            ? 'Camera access was denied. Allow it in your browser settings, or use the Upload tab instead.'
            : err.name === 'NotFoundError'
              ? 'No camera was found on this device.'
              : 'Could not access the camera.';
        setError(message);
      });

    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    };
  }, []);

  useEffect(
    () => () => {
      if (capturedUrl) URL.revokeObjectURL(capturedUrl);
    },
    [capturedUrl],
  );

  function capture() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !video.videoWidth) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    // Drawn un-mirrored, even though the live preview is mirrored — a captured
    // photo should read the same way the scene actually does.
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        if (capturedUrl) URL.revokeObjectURL(capturedUrl);
        const url = URL.createObjectURL(blob);
        setCapturedUrl(url);
        onFile(new File([blob], `camera-${Date.now()}.jpg`, { type: 'image/jpeg' }));
      },
      'image/jpeg',
      0.92,
    );
  }

  function retake() {
    if (capturedUrl) URL.revokeObjectURL(capturedUrl);
    setCapturedUrl(null);
  }

  return (
    <div>
      <div className="camera-frame">
        {error ? (
          <div className="camera-unavailable">
            <CameraIcon style={{ width: 24, height: 24, marginBottom: 8 }} />
            <p>{error}</p>
          </div>
        ) : (
          <>
            <video ref={videoRef} autoPlay playsInline muted style={{ display: capturedUrl ? 'none' : 'block' }} />
            {capturedUrl && <img src={capturedUrl} alt="Captured frame" />}
            {ready && !capturedUrl && (
              <span className="camera-badge">
                <span className="rec-dot" /> live
              </span>
            )}
          </>
        )}
      </div>
      <canvas ref={canvasRef} style={{ display: 'none' }} />
      {!error && (
        <div className="camera-actions">
          {capturedUrl ? (
            <button type="button" className="btn btn-ghost btn-sm" onClick={retake} disabled={disabled}>
              Retake
            </button>
          ) : (
            <button type="button" className="btn btn-primary btn-sm" onClick={capture} disabled={disabled || !ready}>
              Capture frame
            </button>
          )}
        </div>
      )}
    </div>
  );
}
