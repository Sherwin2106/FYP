export default function ExplanationBlock({ explanation }) {
  if (!explanation) return null;
  return (
    <section className="card card-body">
      <h2 className="section-title">Explanation</h2>
      <p className="explanation-text">{explanation}</p>
    </section>
  );
}
