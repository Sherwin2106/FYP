import { useCallback, useEffect, useRef, useState } from 'react';
import Header from './components/Header';
import SourcePanel from './components/SourcePanel';
import HistoryStrip from './components/HistoryStrip';
import ResultsPanel from './components/ResultsPanel';
import { fetchHealth, startAnalysis, startVideoAnalysis, pollJob } from './lib/api';

const HISTORY_KEY = 'svcr_history_v1';
const MAX_HISTORY = 8;
const POLL_INTERVAL_MS = 700;
const HEALTH_INTERVAL_MS = 20000;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Centre-cropped square thumbnail, kept small so session history stays cheap to store. */
function makeThumbnail(file) {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const size = 160;
      const scale = Math.max(size / img.width, size / img.height);
      const w = img.width * scale;
      const h = img.height * scale;
      const canvas = document.createElement('canvas');
      canvas.width = size;
      canvas.height = size;
      canvas.getContext('2d').drawImage(img, (size - w) / 2, (size - h) / 2, w, h);
      URL.revokeObjectURL(url);
      resolve(canvas.toDataURL('image/jpeg', 0.72));
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      resolve(null);
    };
    img.src = url;
  });
}

function summarize(result) {
  if (result?.kind === 'video') {
    const a = result.summary?.activity;
    return a ? `Video · ${a.label}` : 'Video analysis';
  }
  const p = result?.prediction;
  return p ? `${p.activity.label} · ${p.relationship.label}` : 'Analysis';
}

export default function App() {
  const [health, setHealth] = useState(null);

  const [mode, setMode] = useState('upload');
  const [file, setFile] = useState(null);
  const [videoFile, setVideoFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [videoProgress, setVideoProgress] = useState(null);
  const [isVideoRun, setIsVideoRun] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [question, setQuestion] = useState('');
  const [choices, setChoices] = useState([]);

  const [status, setStatus] = useState('idle'); // idle | queued | running | done | error
  const [activeStage, setActiveStage] = useState(null);
  const [doneStages, setDoneStages] = useState([]);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [analyzedAt, setAnalyzedAt] = useState(null);
  const [currentResultId, setCurrentResultId] = useState(null);

  const [history, setHistory] = useState([]);

  // Guards against race conditions: if the user starts a new analysis (or the
  // component unmounts) while an old poll loop is still in flight, that loop's
  // state updates must be ignored instead of clobbering the newer run.
  const runIdRef = useRef(0);
  const mountedRef = useRef(true);

  useEffect(() => {
    // Under React 18 StrictMode, development mounts every component once,
    // tears it down, then mounts it again as a diagnostic — so the setup
    // phase must reset this to true, not just rely on the initial useRef
    // value, or the ref is left permanently false after that first cycle
    // and every analysis silently aborts right after it starts.
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  // -- health polling ------------------------------------------------------
  useEffect(() => {
    let cancelled = false;
    async function check() {
      try {
        const data = await fetchHealth();
        if (!cancelled) setHealth(data);
      } catch {
        if (!cancelled) setHealth({ status: 'error', error: 'The analysis service could not be reached.' });
      }
    }
    check();
    const id = setInterval(check, HEALTH_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  // -- history persistence --------------------------------------------------
  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(HISTORY_KEY);
      if (raw) setHistory(JSON.parse(raw));
    } catch {
      // ignore corrupt/unavailable storage
    }
  }, []);

  useEffect(() => {
    try {
      sessionStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    } catch {
      // storage full or unavailable (e.g. private browsing) — history just won't persist
    }
  }, [history]);

  // -- file selection ---------------------------------------------------------
  const handleFile = useCallback(
    (newFile) => {
      setFile(newFile);
      setPreviewUrl((old) => {
        if (old) URL.revokeObjectURL(old);
        return URL.createObjectURL(newFile);
      });
      setStatus('idle');
      setError(null);
      setResult(null);
      setCurrentResultId(null);
    },
    [],
  );

  useEffect(
    () => () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    },
    [previewUrl],
  );

  const handleVideoFile = useCallback((newFile) => {
    setVideoFile(newFile);
    setVideoUrl((old) => {
      if (old) URL.revokeObjectURL(old);
      return URL.createObjectURL(newFile);
    });
    setStatus('idle');
    setError(null);
    setResult(null);
    setCurrentResultId(null);
  }, []);

  // -- analysis ---------------------------------------------------------------
  const handleAnalyze = useCallback(async () => {
    const video = mode === 'video';
    if (video ? !videoFile : !file) return;
    const runId = (runIdRef.current += 1);
    const isCurrent = () => mountedRef.current && runIdRef.current === runId;

    setStatus('queued');
    setError(null);
    setActiveStage(null);
    setDoneStages([]);
    setIsVideoRun(video);
    setVideoProgress(null);

    let jobId;
    try {
      const thumbPromise = video ? Promise.resolve(null) : makeThumbnail(file);
      const res = video
        ? await startVideoAnalysis({ file: videoFile, question })
        : await startAnalysis({ file, question, choices });
      jobId = res.job_id;
      if (!isCurrent()) return;
      setCurrentResultId(jobId);
      setStatus('running');

      // eslint-disable-next-line no-constant-condition
      while (true) {
        const data = await pollJob(jobId);
        if (!isCurrent()) return;
        setActiveStage(data.stage);
        setDoneStages(data.done_stages || []);
        if (data.kind === 'video') {
          setVideoProgress({ scan: data.scan_progress, moments: data.moments || [], current: data.current });
        }

        if (data.status === 'done') {
          setResult(data.result);
          setStatus('done');
          const when = Date.now();
          setAnalyzedAt(when);
          const thumbnail = video ? data.result.moments?.find((m) => m.thumbnail)?.thumbnail : await thumbPromise;
          if (!isCurrent()) return;
          setHistory((prev) => [
            { id: jobId, thumbnail, result: data.result, analyzedAt: when, summary: summarize(data.result) },
            ...prev.filter((h) => h.id !== jobId),
          ].slice(0, MAX_HISTORY));
          return;
        }
        if (data.status === 'error') {
          setStatus('error');
          setError(data.error || 'The analysis failed for an unknown reason.');
          return;
        }
        await sleep(POLL_INTERVAL_MS);
      }
    } catch (err) {
      if (!isCurrent()) return;
      setStatus('error');
      setError(err.message || 'The analysis service could not be reached. Please try again.');
    }
  }, [mode, file, videoFile, question, choices]);

  const handleSelectHistory = useCallback(
    (id) => {
      const entry = history.find((h) => h.id === id);
      if (!entry) return;
      runIdRef.current += 1; // cancel any in-flight poll before showing a past result
      setStatus('done');
      setResult(entry.result);
      setAnalyzedAt(entry.analyzedAt);
      setCurrentResultId(entry.id);
      setError(null);
    },
    [history],
  );

  return (
    <div className="app">
      <Header health={health} />
      <main className="layout">
        <div className="source-column">
          <SourcePanel
            mode={mode}
            setMode={setMode}
            previewUrl={previewUrl}
            onFile={handleFile}
            videoUrl={videoUrl}
            onVideoFile={handleVideoFile}
            question={question}
            setQuestion={setQuestion}
            choices={choices}
            setChoices={setChoices}
            onAnalyze={handleAnalyze}
            status={status}
            activeStage={activeStage}
            doneStages={doneStages}
            hasFile={Boolean(mode === 'video' ? videoFile : file)}
          />
          <HistoryStrip items={history} activeId={currentResultId} onSelect={handleSelectHistory} />
        </div>

        <div className="results-column">
          <ResultsPanel
            status={status}
            activeStage={activeStage}
            doneStages={doneStages}
            error={error}
            result={result}
            analyzedAt={analyzedAt}
            onRetry={handleAnalyze}
            isVideo={isVideoRun}
            videoProgress={videoProgress}
          />
        </div>
      </main>

      <footer className="site-footer">
        <span>Explainable Visual Commonsense Reasoning for Social Interactions</span>
        <span>Department of Computer Science and Engineering, SSN College of Engineering</span>
      </footer>
    </div>
  );
}
