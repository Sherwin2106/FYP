// A handful of plain geometric line icons, drawn by hand rather than pulled
// from an icon package — keeps the mark language consistent with the rest of
// the UI instead of mixing in a generic rounded icon-font look.

const base = {
  width: '1em',
  height: '1em',
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.6,
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
};

export function ImageIcon(props) {
  return (
    <svg {...base} {...props}>
      <rect x="3" y="4" width="18" height="16" rx="1.5" />
      <circle cx="8.5" cy="9.5" r="1.6" />
      <path d="M4 17l5-5 3.5 3.5L17 10l3 3" />
    </svg>
  );
}

export function CameraIcon(props) {
  return (
    <svg {...base} {...props}>
      <path d="M4 8.5h3l1.3-2h7.4l1.3 2h3v11H4z" />
      <circle cx="12" cy="13.8" r="3.3" />
    </svg>
  );
}

export function ChevronIcon(props) {
  return (
    <svg {...base} {...props}>
      <path d="M9 5l7 7-7 7" />
    </svg>
  );
}

export function CheckIcon(props) {
  return (
    <svg {...base} strokeWidth={2.4} {...props}>
      <path d="M4 12.5l5 5L20 6" />
    </svg>
  );
}

export function PeopleIcon(props) {
  return (
    <svg {...base} {...props}>
      <circle cx="9" cy="8.5" r="3" />
      <path d="M3.5 20c0-3.5 2.5-6 5.5-6s5.5 2.5 5.5 6" />
      <circle cx="17.5" cy="9.5" r="2.3" />
      <path d="M15 20c0-2.6 1.4-4.8 3.4-5.6" transform="translate(0.5 0)" />
    </svg>
  );
}

export function NodeLinkIcon(props) {
  return (
    <svg {...base} {...props}>
      <circle cx="5.5" cy="6" r="2.2" />
      <circle cx="18.5" cy="6" r="2.2" />
      <circle cx="12" cy="18" r="2.2" />
      <path d="M7.4 7.1L10.3 16M16.6 7.1L13.7 16M7.7 6h8.6" />
    </svg>
  );
}

export function SparkIcon(props) {
  return (
    <svg {...base} {...props}>
      <path d="M12 3l1.8 5.6L19.5 10 13.8 11.9 12 17.5l-1.8-5.6L4.5 10l5.7-1.4z" />
    </svg>
  );
}

export function AlertIcon(props) {
  return (
    <svg {...base} {...props}>
      <path d="M12 4l9 15.5H3z" />
      <path d="M12 10.3v3.6" strokeWidth={2} />
      <circle cx="12" cy="17" r="0.9" fill="currentColor" stroke="none" />
    </svg>
  );
}

export function CloseIcon(props) {
  return (
    <svg {...base} {...props}>
      <path d="M6 6l12 12M18 6L6 18" />
    </svg>
  );
}

export function ClockIcon(props) {
  return (
    <svg {...base} {...props}>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M12 7.5V12l3 2" />
    </svg>
  );
}
