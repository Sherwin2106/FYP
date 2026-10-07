import { pct } from '../lib/format';

export default function QAPanel({ prediction }) {
  if (!prediction?.question) return null;
  const { question, answer, answer_choices: choices, answer_probabilities: probs } = prediction;

  return (
    <section className="card card-body">
      <h2 className="section-title">Your question</h2>
      <p className="qa-question">{question}</p>
      <p className="qa-answer">{answer || 'No answer was produced.'}</p>

      {choices && probs && (
        <ul className="qa-choices">
          {choices.map((choice, i) => {
            const chosen = choice === answer;
            return (
              <li key={i}>
                <div className={`qa-choice-top${chosen ? ' chosen' : ''}`}>
                  <span>{choice}</span>
                  <span className="num">{pct(probs[i])}</span>
                </div>
                <div className="confidence-track">
                  <div
                    className={`confidence-fill${chosen ? '' : ' warn'}`}
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
