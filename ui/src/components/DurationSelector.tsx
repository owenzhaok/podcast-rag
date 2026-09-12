interface DurationSelectorProps {
  value: number;
  onChange: (v: number) => void;
}

const OPTIONS = [1, 2, 5, 10];

export default function DurationSelector({ value, onChange }: DurationSelectorProps) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      className="border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
    >
      {OPTIONS.map((opt) => (
        <option key={opt} value={opt}>
          {opt} min
        </option>
      ))}
    </select>
  );
}
