export default function ExplanationBlock({ explanation }) {
  if (!explanation) return null;
  return (
    <section className="card explanation-block">
      <p className="eyebrow">Explanation</p>
      <blockquote>{explanation}</blockquote>
    </section>
  );
}
