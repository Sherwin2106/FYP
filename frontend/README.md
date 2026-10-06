# Social VCR — React dashboard

The web UI for the Phase I pipeline (see the project root [README](../README.md) for the full
picture, and [backend/main.py](../backend/main.py) for the API it talks to).

```bash
npm install
npm run dev        # http://localhost:5173, proxies /api to http://127.0.0.1:8000
```

The backend must be running separately (`uvicorn backend.main:app --port 8000` from the project
root) — the dev server only proxies to it, it doesn't start it.

## Structure

```
src/
  App.jsx                  state + orchestration (upload/camera, polling, history)
  components/
    SourcePanel.jsx        tabs, dropzone/camera, question form, analyze button + stepper
    Dropzone.jsx            drag-and-drop image upload with validation
    CameraCapture.jsx       live camera preview + frame capture (releases the camera on unmount)
    QuestionForm.jsx        optional free-form question + multiple-choice options
    ProgressStepper.jsx     live module 1→4 progress, driven by the backend's job status
    ResultsPanel.jsx        picks between empty/working/error/dashboard states
    TaskCard.jsx            activity/relationship/intention card with confidence bar
    ExplanationBlock.jsx    the generated textual explanation
    CommonsensePanel.jsx    ConceptNet evidence behind the final prediction
    QAPanel.jsx             answer to an optional user question
    VisualConceptsPanel.jsx what SmolVLM saw (caption, people, actions, objects, gestures)
    TimingsFooter.jsx       per-module timings
    RawJsonViewer.jsx       collapsible raw JSON of the full Phase1Result
    HistoryStrip.jsx        past analyses from this session (sessionStorage)
    StatePanels.jsx         empty / error / "working" placeholders
    Icons.jsx               small hand-drawn inline icon set
  lib/
    api.js                  fetch wrappers for /api/health, /api/analyze, /api/jobs/:id
    format.js                percentage/label/timing formatting helpers
  styles/
    globals.css              palette (CSS variables), typography, resets, primitives
    layout.css                header / two-column grid / footer
    components.css            everything else
```

## Design notes

* **Palette** is fixed as CSS variables in `styles/globals.css` — warm beige/brown, matching the
  project's reference palette. There's no dark mode; the palette was specified as the single
  theme.
* **Type**: Fraunces (display/serif headings), Karla (body), Space Mono (labels, numbers, raw
  JSON) — loaded from Google Fonts in `index.html`.
* **Progress is real**, not simulated: the backend reports which of the four report modules
  (preprocess → SmolVLM → ConceptNet → reasoning) is currently running, and the stepper reflects
  exactly that.
* State updates from a stale analysis run (e.g. the user starts a second analysis before the
  first one's poll loop finishes) are dropped via a run-id guard in `App.jsx`, so a slow old
  request can never overwrite a newer result.
