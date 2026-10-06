export default function RawJsonViewer({ data }) {
  if (!data) return null;
  return (
    <details className="card raw-json">
      <summary>View raw Phase I output (JSON)</summary>
      <pre className="scrollbar-thin">{JSON.stringify(data, null, 2)}</pre>
    </details>
  );
}
