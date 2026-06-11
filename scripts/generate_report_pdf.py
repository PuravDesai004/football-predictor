"""
Football Predictor Model - Report PDF Generator

Usage:
    python scripts/generate_report_pdf.py

Inputs:
    docs/football_predictor_report.md

Outputs:
    docs/Football_Predictor_Report.html
    ~/Downloads/Football_Predictor_Report.pdf  (when Chrome or Edge is available)

The script converts the Markdown report into a styled HTML document and then
uses a local Chromium-based browser, when available, to export a PDF directly
into the user's Downloads folder.
"""

from __future__ import annotations

import html
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_MD_PATH = PROJECT_ROOT / "docs" / "football_predictor_report.md"
OUTPUT_HTML_PATH = PROJECT_ROOT / "docs" / "Football_Predictor_Report.html"
OUTPUT_PDF_PATH = Path.home() / "Downloads" / "Football_Predictor_Report.pdf"


CSS = """
:root {
    --text: #172033;
    --muted: #4b5875;
    --accent: #1f5fbf;
    --accent-soft: #e8f0ff;
    --border: #d7deea;
    --panel: #f8fafd;
    --header: #14213d;
    --warning-bg: #fff6df;
    --warning-border: #d69e2e;
}

* {
    box-sizing: border-box;
}

@page {
    size: A4;
    margin: 1.8cm 2cm;
}

@media print {
    body {
        padding: 0;
        max-width: none;
    }

    .cover-page {
        break-after: page;
    }

    h1, h2, h3 {
        break-after: avoid;
    }

    table, pre, blockquote {
        break-inside: avoid;
    }

    .no-print {
        display: none;
    }
}

body {
    margin: 0 auto;
    padding: 1.8cm 2cm;
    max-width: 210mm;
    font-family: Arial, Helvetica, sans-serif;
    font-size: 10.6pt;
    line-height: 1.62;
    color: var(--text);
    background: #ffffff;
}

.cover-page {
    min-height: 88vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    padding: 2rem;
}

.cover-page h1 {
    font-size: 34pt;
    line-height: 1.12;
    color: var(--header);
    margin: 0 0 0.8rem;
}

.cover-subtitle {
    max-width: 560px;
    font-size: 13.5pt;
    color: var(--muted);
    margin: 0.7rem auto 2rem;
}

.cover-divider {
    width: 72px;
    height: 4px;
    background: var(--accent);
    border-radius: 999px;
    margin: 1.2rem auto 1.6rem;
}

.cover-meta {
    color: var(--muted);
    margin-top: 0.6rem;
}

.cover-tiers {
    margin-top: 2rem;
    padding: 1.2rem 1.5rem;
    border: 1px solid var(--border);
    border-left: 5px solid var(--accent);
    border-radius: 8px;
    background: var(--panel);
    text-align: left;
}

.cover-tiers div {
    margin: 0.35rem 0;
}

h1 {
    color: var(--header);
    font-size: 24pt;
    margin: 2rem 0 1rem;
}

h2 {
    color: var(--header);
    font-size: 17pt;
    margin: 2rem 0 0.9rem;
    padding-bottom: 0.35rem;
    border-bottom: 2px solid var(--accent);
}

h3 {
    color: var(--text);
    font-size: 13pt;
    margin: 1.45rem 0 0.55rem;
}

p {
    margin: 0.55rem 0;
}

ul, ol {
    margin: 0.55rem 0 0.8rem 1.35rem;
    padding: 0;
}

li {
    margin: 0.2rem 0;
}

a {
    color: var(--accent);
}

blockquote {
    margin: 1rem 0;
    padding: 0.85rem 1rem;
    background: var(--warning-bg);
    border-left: 4px solid var(--warning-border);
    border-radius: 0 8px 8px 0;
    color: #6b4c12;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 0.9rem 0 1.25rem;
    font-size: 9.4pt;
}

th {
    background: var(--header);
    color: #ffffff;
    text-align: left;
    padding: 0.55rem 0.65rem;
    font-weight: 700;
}

td {
    border-bottom: 1px solid var(--border);
    padding: 0.48rem 0.65rem;
    vertical-align: top;
}

tr:nth-child(even) td {
    background: var(--panel);
}

code {
    font-family: Consolas, "Courier New", monospace;
    background: #eef2f7;
    color: #174ea6;
    padding: 0.12rem 0.28rem;
    border-radius: 4px;
    font-size: 0.9em;
}

pre {
    background: #f3f6fa;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.9rem 1rem;
    overflow-x: auto;
    white-space: pre-wrap;
}

pre code {
    background: transparent;
    color: var(--text);
    padding: 0;
}

hr {
    border: 0;
    border-top: 1px solid var(--border);
    margin: 2rem 0;
}

.report-footer {
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    text-align: center;
    color: var(--muted);
    font-size: 9pt;
}

.print-btn {
    position: fixed;
    right: 24px;
    bottom: 24px;
    border: 0;
    border-radius: 8px;
    background: var(--accent);
    color: #fff;
    padding: 0.75rem 1.1rem;
    font-family: Arial, Helvetica, sans-serif;
    font-weight: 700;
    box-shadow: 0 6px 18px rgba(31, 95, 191, 0.28);
    cursor: pointer;
}
"""


COVER_HTML = """
<section class="cover-page">
    <h1>Football Predictor Model</h1>
    <div class="cover-divider"></div>
    <p class="cover-subtitle">
        A three-tier Premier League match prediction and Fantasy Premier League optimization project
    </p>
    <div class="cover-meta"><strong>Author:</strong> Purav Desai</div>
    <div class="cover-tiers">
        <div><strong>Tier 1:</strong> Foundation</div>
        <div><strong>Tier 2:</strong> Advanced local ML system</div>
        <div><strong>Tier 3:</strong> Multi-season research and production-ready local pipeline</div>
    </div>
    <div class="cover-meta">June 2026</div>
</section>
"""


def convert_markdown(md_text: str) -> str:
    """Convert Markdown to HTML using the markdown package when available."""
    try:
        import markdown

        body = markdown.markdown(
            md_text,
            extensions=["tables", "fenced_code", "toc", "sane_lists"],
            output_format="html5",
        )
        print("Using markdown package for HTML conversion.")
        return body
    except ImportError:
        print("WARNING: markdown package not found. Using simple fallback converter.")
        return fallback_markdown(md_text)


def fallback_markdown(md_text: str) -> str:
    """Small fallback converter for headings, paragraphs, code blocks, lists, and tables."""
    blocks: list[str] = []
    in_code = False
    code_lines: list[str] = []
    paragraph: list[str] = []
    list_items: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            text = " ".join(paragraph).strip()
            blocks.append(f"<p>{inline_format(text)}</p>")
            paragraph = []

    def flush_list() -> None:
        nonlocal list_items
        if list_items:
            blocks.append("<ul>" + "".join(f"<li>{inline_format(item)}</li>" for item in list_items) + "</ul>")
            list_items = []

    def parse_table(lines: list[str], start: int) -> tuple[str | None, int]:
        if start + 1 >= len(lines):
            return None, start
        header = lines[start]
        sep = lines[start + 1]
        if "|" not in header or not re.match(r"^\s*\|?[\s\-:|]+\|?\s*$", sep):
            return None, start
        table_lines = [header]
        i = start + 2
        while i < len(lines) and "|" in lines[i] and lines[i].strip():
            table_lines.append(lines[i])
            i += 1
        headers = [c.strip() for c in header.strip("|").split("|")]
        html_rows = ["<table><thead><tr>"]
        html_rows.extend(f"<th>{inline_format(c)}</th>" for c in headers)
        html_rows.append("</tr></thead><tbody>")
        for row in table_lines[1:]:
            cells = [c.strip() for c in row.strip("|").split("|")]
            html_rows.append("<tr>")
            html_rows.extend(f"<td>{inline_format(c)}</td>" for c in cells)
            html_rows.append("</tr>")
        html_rows.append("</tbody></table>")
        return "".join(html_rows), i - 1

    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            if not in_code:
                flush_paragraph()
                flush_list()
                in_code = True
                code_lines = []
            else:
                blocks.append("<pre><code>" + html.escape("\n".join(code_lines)) + "</code></pre>")
                in_code = False
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        table_html, new_i = parse_table(lines, i)
        if table_html:
            flush_paragraph()
            flush_list()
            blocks.append(table_html)
            i = new_i + 1
            continue

        if not stripped:
            flush_paragraph()
            flush_list()
            i += 1
            continue

        if stripped.startswith("#"):
            flush_paragraph()
            flush_list()
            level = min(len(stripped) - len(stripped.lstrip("#")), 3)
            title = stripped[level:].strip()
            blocks.append(f"<h{level}>{inline_format(title)}</h{level}>")
        elif stripped.startswith("- "):
            flush_paragraph()
            list_items.append(stripped[2:].strip())
        elif stripped.startswith(">"):
            flush_paragraph()
            flush_list()
            blocks.append(f"<blockquote>{inline_format(stripped[1:].strip())}</blockquote>")
        else:
            paragraph.append(stripped)

        i += 1

    flush_paragraph()
    flush_list()
    return "\n".join(blocks)


def inline_format(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    return escaped


def build_html(md_text: str) -> str:
    body_html = convert_markdown(md_text)
    if len(re.sub(r"<[^>]+>", "", body_html).strip()) < 500:
        raise RuntimeError("Generated HTML body looks too small. Check the Markdown input.")

    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Football Predictor Model - Final Project Report</title>
    <style>{CSS}</style>
</head>
<body>
{COVER_HTML}
<main>
{body_html}
</main>
<footer class="report-footer">
    <p>Football Predictor Model - Final Project Report</p>
    <p>Generated June 2026</p>
</footer>
<button class="print-btn no-print" onclick="window.print()">Print / Save as PDF</button>
</body>
</html>
"""


def find_browser() -> Path | None:
    candidates = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def render_pdf(html_path: Path, pdf_path: Path) -> bool:
    browser = find_browser()
    if browser is None:
        print("WARNING: Chrome/Edge was not found. HTML was generated, but PDF export was skipped.")
        return False

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if pdf_path.exists():
        pdf_path.unlink()

    command = [
        str(browser),
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--allow-file-access-from-files",
        f"--print-to-pdf={pdf_path}",
        html_path.resolve().as_uri(),
    ]
    print(f"Rendering PDF with: {browser}")
    subprocess.run(command, check=True)

    if not pdf_path.exists() or pdf_path.stat().st_size < 10_000:
        raise RuntimeError(f"PDF export failed or produced a tiny file: {pdf_path}")
    return True


def main() -> int:
    print("Football Predictor Model - Report Generator")
    print("=" * 52)

    if not REPORT_MD_PATH.exists():
        print(f"ERROR: Markdown report not found: {REPORT_MD_PATH}")
        return 1

    md_text = REPORT_MD_PATH.read_text(encoding="utf-8")
    if len(md_text.strip()) < 500:
        print("ERROR: Markdown report is too small to be the final project report.")
        return 1

    OUTPUT_HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    html_text = build_html(md_text)
    OUTPUT_HTML_PATH.write_text(html_text, encoding="utf-8")
    print(f"HTML written: {OUTPUT_HTML_PATH}")
    print(f"HTML size: {OUTPUT_HTML_PATH.stat().st_size:,} bytes")

    try:
        rendered = render_pdf(OUTPUT_HTML_PATH, OUTPUT_PDF_PATH)
    except Exception as exc:
        print(f"WARNING: PDF rendering failed: {exc}")
        rendered = False

    if rendered:
        print(f"PDF written: {OUTPUT_PDF_PATH}")
        print(f"PDF size: {OUTPUT_PDF_PATH.stat().st_size:,} bytes")
    else:
        print("Open the HTML file in Chrome and use Print -> Save as PDF.")
        print(f"HTML file: {OUTPUT_HTML_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
