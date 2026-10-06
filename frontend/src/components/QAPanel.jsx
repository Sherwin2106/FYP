import { pct } from '../lib/format';

export default function QAPanel({ prediction }) {
  if (!prediction?.question) return null;
  const { question, answer, answer_choices: choices, answer_probabilities: probs } = prediction;

  return (
    <section className="card qa-panel">
      <p className="eyebrow">Your question</p>
      <p className="qa-question">“{question}”</p>
      <p className="qa-answer">{answer || '—'}</p>

      {choices && probs && (
        <ul className="qa-choices">
          {choices.map((choice, i) => {
            const chosen = choice === answer;
            return (
              <li className="qa-choice-row" key={i}>
                <div className={`qa-choice-top${chosen ? ' chosen' : ''}`}>
                  <span>{choice}</span>
                  <span className="mono">{pct(probs[i])}</span>
                </div>
                <div className="confidence-track">
                  <div
                    className={`confidence-fill${chosen ? '' : ' low'}`}
                    style={{ width: `${Math.max(3, Math.round(probs[i] * 100))}%` }}
                  />
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
