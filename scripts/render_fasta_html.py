from __future__ import annotations

import argparse
import gzip
import html
from pathlib import Path
from typing import Iterable


BASE_CLASS = {
    "A": "base-a",
    "C": "base-c",
    "G": "base-g",
    "T": "base-t",
    "U": "base-u",
}

AMBIGUOUS_BASES = set("NRYKMSWBDHVX")


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_fasta(path: Path) -> Iterable[tuple[str, str]]:
    header: str | None = None
    chunks: list[str] = []
    with open_text(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks).upper()
                header = line[1:].strip()
                chunks = []
                continue
            chunks.append(line)

    if header is not None:
        yield header, "".join(chunks).upper()


def gc_fraction(sequence: str) -> float:
    gc = sequence.count("G") + sequence.count("C")
    acgtu = sum(sequence.count(base) for base in "ACGTU")
    return (gc / acgtu) if acgtu else 0.0


def non_acgt_fraction(sequence: str) -> float:
    if not sequence:
        return 0.0
    regular = sum(sequence.count(base) for base in "ACGTU")
    return (len(sequence) - regular) / len(sequence)


def repeat_counts(sequence: str, kmer_size: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    if len(sequence) < kmer_size:
        return counts

    for index in range(len(sequence) - kmer_size + 1):
        kmer = sequence[index : index + kmer_size]
        if any(base not in "ACGTU" for base in kmer):
            continue
        counts[kmer] = counts.get(kmer, 0) + 1
    return counts


def top_repeated_kmers(sequence: str, kmer_size: int, min_count: int, limit: int) -> list[tuple[str, int]]:
    counts = repeat_counts(sequence, kmer_size)
    repeated = [(kmer, count) for kmer, count in counts.items() if count >= min_count]
    repeated.sort(key=lambda item: (-item[1], item[0]))
    return repeated[:limit]


def build_repeat_mask(display_sequence: str, repeated_kmers: list[tuple[str, int]], kmer_size: int) -> list[int]:
    mask = [0] * len(display_sequence)
    if not repeated_kmers:
        return mask

    repeated_lookup = {kmer: count for kmer, count in repeated_kmers}
    for index in range(max(0, len(display_sequence) - kmer_size + 1)):
        kmer = display_sequence[index : index + kmer_size]
        count = repeated_lookup.get(kmer)
        if count is None:
            continue
        level = 2 if count >= 8 else 1
        for offset in range(kmer_size):
            pos = index + offset
            if pos < len(mask):
                mask[pos] = max(mask[pos], level)
    return mask


def render_sequence_html(
    sequence: str,
    start_base: int,
    max_bases: int,
    line_width: int,
    kmer_size: int,
    min_repeat_count: int,
    top_kmers_limit: int,
) -> tuple[str, str]:
    zero_based_start = max(0, start_base - 1)
    if max_bases <= 0:
        display_sequence = sequence[zero_based_start:]
    else:
        display_sequence = sequence[zero_based_start : zero_based_start + max_bases]
    repeated_kmers = top_repeated_kmers(display_sequence, kmer_size, min_repeat_count, top_kmers_limit)
    repeat_mask = build_repeat_mask(display_sequence, repeated_kmers, kmer_size)

    line_blocks: list[str] = []
    for line_start in range(0, len(display_sequence), line_width):
        line_end = min(line_start + line_width, len(display_sequence))
        chunk = display_sequence[line_start:line_end]
        chunk_mask = repeat_mask[line_start:line_end]
        absolute_start = zero_based_start + line_start + 1
        absolute_end = zero_based_start + line_end
        repeat_fraction = (sum(1 for value in chunk_mask if value > 0) / len(chunk)) if chunk else 0.0

        rendered_chunk: list[str] = []
        for base, repeat_level in zip(chunk, chunk_mask):
            classes = [BASE_CLASS.get(base, "base-amb")]
            if base in AMBIGUOUS_BASES:
                classes = ["base-amb"]
            if repeat_level == 1:
                classes.append("repeat-soft")
            elif repeat_level >= 2:
                classes.append("repeat-strong")
            rendered_chunk.append(f'<span class="{" ".join(classes)}">{html.escape(base)}</span>')

        meter_width = max(4, int(repeat_fraction * 100))
        line_blocks.append(
            (
                '<div class="seq-line">'
                f'<span class="coord">{absolute_start:>12,} - {absolute_end:<12,}</span>'
                f'<span class="repeat-meter"><span class="repeat-meter-fill" style="width:{meter_width}%"></span></span>'
                f'<code class="bases">{"".join(rendered_chunk)}</code>'
                "</div>"
            )
        )

    repeat_summary = "<li>None above threshold in displayed region</li>"
    if repeated_kmers:
        repeat_summary = "".join(
            f"<li><code>{html.escape(kmer)}</code> x {count}</li>" for kmer, count in repeated_kmers
        )

    return "\n".join(line_blocks), repeat_summary


def render_report(
    fasta_path: Path,
    output_path: Path,
    start_base: int,
    window_size: int,
    line_width: int,
    kmer_size: int,
    min_repeat_count: int,
    top_kmers_limit: int,
) -> None:
    sequence_sections: list[str] = []
    total_sequences = 0
    total_bases = 0

    for header, sequence in iter_fasta(fasta_path):
        total_sequences += 1
        total_bases += len(sequence)
        rendered_lines, repeated_kmers = render_sequence_html(
            sequence=sequence,
            start_base=start_base,
            max_bases=window_size,
            line_width=line_width,
            kmer_size=kmer_size,
            min_repeat_count=min_repeat_count,
            top_kmers_limit=top_kmers_limit,
        )
        shown_end = len(sequence) if window_size <= 0 else min(len(sequence), start_base - 1 + window_size)
        truncated_note = ""
        if shown_end < len(sequence):
            truncated_note = (
                f"<p>Showing a single window from base <code>{start_base:,}</code> to "
                f"<code>{shown_end:,}</code>. Re-run with a different <code>--start-base</code> "
                f"or larger <code>--window-size</code> to inspect another region.</p>"
            )
        sequence_sections.append(
            f"""
            <section class="record">
              <h2>{html.escape(header)}</h2>
              <div class="meta-grid">
                <div><span class="meta-label">Length</span><span class="meta-value">{len(sequence):,}</span></div>
                <div><span class="meta-label">GC</span><span class="meta-value">{gc_fraction(sequence) * 100:.2f}%</span></div>
                <div><span class="meta-label">Non-ACGT</span><span class="meta-value">{non_acgt_fraction(sequence) * 100:.2f}%</span></div>
                <div><span class="meta-label">Shown</span><span class="meta-value">{start_base:,}..{shown_end:,}</span></div>
              </div>
              {truncated_note}
              <div class="subpanel">
                <h3>Top repeated {kmer_size}-mers in displayed region</h3>
                <ul class="repeat-list">{repeated_kmers}</ul>
              </div>
              <div class="subpanel">
                <h3>Sequence view</h3>
                <div class="legend">
                  <span class="legend-chip base-a">A</span>
                  <span class="legend-chip base-c">C</span>
                  <span class="legend-chip base-g">G</span>
                  <span class="legend-chip base-t">T</span>
                  <span class="legend-chip base-amb">N/ambiguous</span>
                  <span class="legend-chip repeat-soft">repeat</span>
                  <span class="legend-chip repeat-strong">strong repeat</span>
                </div>
                <div class="sequence-block">{rendered_lines}</div>
              </div>
            </section>
            """
        )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>FASTA Viewer - {html.escape(fasta_path.name)}</title>
  <style>
    :root {{
      --bg: #0f1115;
      --panel: #171a21;
      --panel-2: #1e2430;
      --text: #e7edf3;
      --muted: #94a3b8;
      --accent: #f59e0b;
      --border: #2b3444;
      --a: #ff6b6b;
      --c: #4dabf7;
      --g: #ffd43b;
      --t: #69db7c;
      --u: #69db7c;
      --amb: #adb5bd;
      --repeat-soft: rgba(255, 224, 102, 0.24);
      --repeat-strong: rgba(255, 146, 43, 0.40);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #0b0d12, var(--bg));
      color: var(--text);
    }}
    .page {{
      width: min(1500px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 24px 0 48px;
    }}
    .hero, .record {{
      background: color-mix(in srgb, var(--panel) 92%, #000 8%);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 20px 22px;
      margin-bottom: 20px;
      box-shadow: 0 18px 40px rgba(0, 0, 0, 0.28);
    }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    h1 {{ font-size: 28px; }}
    h2 {{ font-size: 20px; word-break: break-word; }}
    h3 {{ font-size: 14px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }}
    p, li {{ color: var(--muted); }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
      margin: 14px 0 18px;
    }}
    .meta-grid > div, .hero-grid > div {{
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px 14px;
    }}
    .hero-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin-top: 14px;
    }}
    .meta-label {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
    }}
    .meta-value {{ font-size: 20px; font-weight: 700; }}
    .subpanel {{
      margin-top: 14px;
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 14px;
    }}
    .repeat-list {{
      margin: 0;
      padding-left: 18px;
      columns: 2 220px;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 12px;
    }}
    .legend-chip {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      padding: 4px 10px;
      border: 1px solid var(--border);
      background: #11151d;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      color: var(--text);
    }}
    .sequence-block {{
      overflow-x: auto;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: #0d1016;
      padding: 10px;
    }}
    .seq-line {{
      display: grid;
      grid-template-columns: 180px 88px 1fr;
      gap: 12px;
      align-items: center;
      margin-bottom: 4px;
    }}
    .coord {{
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }}
    .repeat-meter {{
      position: relative;
      height: 8px;
      border-radius: 999px;
      background: #1d2330;
      overflow: hidden;
      border: 1px solid #293142;
    }}
    .repeat-meter-fill {{
      position: absolute;
      inset: 0 auto 0 0;
      background: linear-gradient(90deg, #ffd43b, #ff922b);
    }}
    .bases {{
      white-space: pre;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 13px;
      letter-spacing: 0.04em;
      line-height: 1.55;
    }}
    .base-a {{ color: var(--a); }}
    .base-c {{ color: var(--c); }}
    .base-g {{ color: var(--g); }}
    .base-t {{ color: var(--t); }}
    .base-u {{ color: var(--u); }}
    .base-amb {{ color: var(--amb); }}
    .repeat-soft {{ background: var(--repeat-soft); border-radius: 2px; }}
    .repeat-strong {{ background: var(--repeat-strong); border-radius: 2px; }}
    code {{
      color: #ffe08a;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <h1>{html.escape(fasta_path.name)}</h1>
      <p>
        Local FASTA view with nucleotide coloring, header separation, simple repeat-density hints,
        and repeated {kmer_size}-mer counts for the displayed region.
      </p>
      <div class="hero-grid">
        <div><span class="meta-label">Sequences</span><span class="meta-value">{total_sequences:,}</span></div>
        <div><span class="meta-label">Total bases</span><span class="meta-value">{total_bases:,}</span></div>
        <div><span class="meta-label">Display window</span><span class="meta-value">{start_base:,} + {'full' if window_size <= 0 else f'{window_size:,}'}</span></div>
        <div><span class="meta-label">Repeat heuristic</span><span class="meta-value">{kmer_size}-mer x {min_repeat_count}+</span></div>
      </div>
    </section>
    {''.join(sequence_sections)}
  </main>
</body>
</html>
"""

    output_path.write_text(page, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render FASTA into a local colorized HTML report.")
    parser.add_argument("input", type=Path, help="Path to a FASTA file (.fa/.fasta or .gz).")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output HTML path. Defaults to replacing the file suffix with .html.",
    )
    parser.add_argument("-s", "--start-base", type=int, default=1, help="1-based start position for rendering.")
    parser.add_argument(
        "-w",
        "--window-size",
        type=int,
        default=50000,
        help="Number of bases to render from the start position for each record. Use 0 for full sequence.",
    )
    parser.add_argument("--line-width", type=int, default=120, help="Bases per rendered line.")
    parser.add_argument("--kmer-size", type=int, default=12, help="K-mer size for repeat hints.")
    parser.add_argument(
        "--min-repeat-count",
        type=int,
        default=4,
        help="Minimum displayed-region repeat count for a k-mer to be highlighted.",
    )
    parser.add_argument(
        "--top-kmers",
        type=int,
        default=12,
        help="Number of repeated k-mers to list in the report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve() if args.output else input_path.with_suffix(".html")
    render_report(
        fasta_path=input_path,
        output_path=output_path,
        start_base=args.start_base,
        window_size=args.window_size,
        line_width=args.line_width,
        kmer_size=args.kmer_size,
        min_repeat_count=args.min_repeat_count,
        top_kmers_limit=args.top_kmers,
    )
    print(output_path)


if __name__ == "__main__":
    main()
