import { useState } from 'react';
import { ChevronIcon, CloseIcon } from './Icons';

const MAX_CHOICES = 6;

export default function QuestionForm({ question, setQuestion, choices, setChoices, disabled }) {
  const [open, setOpen] = useState(false);

  function updateChoice(i, value) {
    const next = [...choices];
    next[i] = value;
    setChoices(next);
  }

  function removeChoice(i) {
    setChoices(choices.filter((_, idx) => idx !== i));
  }

  function addChoice() {
    if (choices.length < MAX_CHOICES) setChoices([...choices, '']);
  }

  return (
    <div className="question-form">
      <button type="button" className="question-toggle" aria-expanded={open} onClick={() => setOpen((v) => !v)}>
        <ChevronIcon />
        Ask a question about the scene <span className="muted">(optional)</span>
      </button>
      {open && (
        <div className="question-body">
          <label className="field-label" htmlFor="question-input">
            Question
          </label>
          <textarea
            id="question-input"
            className="text-input"
            placeholder="e.g. Why are they shaking hands?"
            value={question}
            disabled={disabled}
            onChange={(e) => setQuestion(e.target.value)}
          />

          <div style={{ marginTop: 14 }}>
            <label className="field-label">Multiple-choice answers (optional — leave empty for a free-form answer)</label>
            <div className="choice-list">
              {choices.map((c, i) => (
                <div className="choice-row" key={i}>
                  <input
                    className="text-input"
                    type="text"
                    placeholder={`Option ${i + 1}`}
                    value={c}
                    disabled={disabled}
                    onChange={(e) => updateChoice(i, e.target.value)}
                  />
                  <button
                    type="button"
                    className="choice-remove"
                    aria-label={`Remove option ${i + 1}`}
                    onClick={() => removeChoice(i)}
                    disabled={disabled}
                  >
                    <CloseIcon style={{ width: 13, height: 13 }} />
                  </button>
                </div>
              ))}
            </div>
            {choices.length < MAX_CHOICES && (
              <button type="button" className="add-choice" onClick={addChoice} disabled={disabled}>
                + add option
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
