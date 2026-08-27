"""Render a Markdown document to a print-ready HTML page, for saving as PDF.

    python scripts/make_pdf.py docs/solution.md
    # then open the generated .html and press Ctrl+P -> "Save as PDF"

Deliberately *not* a direct PDF writer. WeasyPrint needs a GTK toolchain on
Windows, and ReportLab means hand-laying every paragraph. The browser already
has an excellent, correctly-hyphenating, font-embedding PDF engine, and every
machine has one. This script's whole job is to hand it a page with print CSS
that paginates sensibly: A4 margins, no orphaned headings, code blocks that do
not split across a page boundary, and links printed as plain text.

Page count is what the brief actually constrains (4-8 pages), so the script
reports an estimate and warns when the document is likely outside that range.
"""

from __future__ import annotations

import argparse
import re
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Rough words-per-page at the stylesheet below. Calibrated against a page of
#: mixed prose, tables and code; treat it as a sanity check, not a measurement.
WORDS_PER_PAGE = 520

STYLESHEET = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }

html {
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
  /* This is a printed document, so it is light regardless of the reader's
     browser theme. Without these, a dark-mode browser renders dark text on a
     dark ground and the "Save as PDF" preview is unreadable. */
  color-scheme: light;
  background: #ffffff;
}

body {
  background: #ffffff;
  font-family: "Charter", "Georgia", "Times New Roman", serif;
  font-size: 10.5pt;
  line-height: 1.5;
  color: #16181d;
  max-width: 46em;
  margin: 0 auto;
  padding: 2em 1em;
  hyphens: auto;
}

h1, h2, h3, h4 {
  font-family: "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  line-height: 1.25;
  color: #0b0d10;
  /* A heading alone at the foot of a page is the most common ugly break. */
  break-after: avoid-page;
  page-break-after: avoid;
}
h1 { font-size: 20pt; margin: 0 0 0.2em; letter-spacing: -0.01em; }
h2 {
  font-size: 14pt;
  margin: 1.9em 0 0.5em;
  padding-bottom: 0.25em;
  border-bottom: 1px solid #d7dbe0;
}
h3 { font-size: 11.5pt; margin: 1.4em 0 0.35em; }
h4 { font-size: 10.5pt; margin: 1.1em 0 0.3em; color: #3d444d; }

p, li { orphans: 3; widows: 3; }
ul, ol { padding-left: 1.3em; }
li { margin: 0.2em 0; }

blockquote {
  margin: 1em 0;
  padding: 0.55em 1em;
  border-left: 3px solid #9aa4b2;
  background: #f5f7f9;
  color: #2c3138;
}
blockquote p { margin: 0.3em 0; }

code {
  font-family: "JetBrains Mono", "Cascadia Mono", "Consolas", monospace;
  font-size: 0.86em;
  background: #f0f2f5;
  padding: 0.1em 0.32em;
  border-radius: 3px;
}

pre {
  background: #f5f7f9;
  border: 1px solid #dfe3e8;
  border-radius: 4px;
  padding: 0.7em 0.9em;
  overflow-x: auto;
  font-size: 8.6pt;
  line-height: 1.42;
  /* Splitting a transcript across a page break makes it unreadable. */
  break-inside: avoid-page;
  page-break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: inherit; }

table {
  border-collapse: collapse;
  width: 100%;
  margin: 1em 0;
  font-size: 9pt;
  break-inside: avoid-page;
  page-break-inside: avoid;
}
th, td { border: 1px solid #d7dbe0; padding: 0.38em 0.55em; text-align: left; vertical-align: top; }
th { background: #eef1f4; font-weight: 600; }

hr { border: none; border-top: 1px solid #d7dbe0; margin: 2em 0; }

a { color: #16181d; text-decoration: none; border-bottom: 1px solid #b9c0c9; }

@media print {
  body { padding: 0; max-width: none; }
  /* A printed page cannot be clicked, so show where a link actually went --
     but only for external ones; in-document anchors would be noise. */
  a[href^="http"]::after { content: " (" attr(href) ")"; font-size: 8pt; color: #5b636d; }
  .no-print { display: none; }
}

.hint {
  font-family: "Inter", "Segoe UI", sans-serif;
  font-size: 9pt;
  background: #fff8e1;
  border: 1px solid #f0d999;
  border-radius: 4px;
  padding: 0.7em 1em;
  margin-bottom: 2em;
}
"""

HINT = (
    '<div class="hint no-print"><strong>To save as PDF:</strong> press '
    "<kbd>Ctrl</kbd>+<kbd>P</kbd> (<kbd>&#8984;</kbd>+<kbd>P</kbd> on macOS), choose "
    '"Save as PDF", set margins to <em>Default</em> and enable background graphics. '
    "This banner does not print.</div>"
)

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
{hint}
{body}
</body>
</html>
"""


def render(source: Path, *, out: Path | None = None) -> Path:
    """Convert one Markdown file to a print-ready HTML file next to it."""
    try:
        import markdown
    except ImportError:  # pragma: no cover - environment problem, not logic
        print(
            "error: the `markdown` package is required.\n"
            "       pip install -r requirements-dev.txt",
            file=sys.stderr,
        )
        raise SystemExit(2) from None

    text = source.read_text(encoding="utf-8")
    body = markdown.markdown(
        text,
        extensions=["extra", "sane_lists", "toc"],
        output_format="html5",
    )

    title = next(
        (line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("# ")),
        source.stem,
    )
    destination = out or source.with_suffix(".html")
    destination.write_text(
        TEMPLATE.format(title=title, css=STYLESHEET, hint=HINT, body=body), encoding="utf-8"
    )
    return destination


def estimate_pages(source: Path) -> int:
    """Words outside code fences, converted to a page count."""
    text = re.sub(r"```.*?```", " ", source.read_text(encoding="utf-8"), flags=re.DOTALL)
    return max(1, round(len(text.split()) / WORDS_PER_PAGE))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", help="the Markdown file to render")
    parser.add_argument("-o", "--out", help="output .html path (default: alongside the source)")
    parser.add_argument("--open", action="store_true", help="open it in the default browser")
    parser.add_argument(
        "--min-pages", type=int, default=4, help="warn below this estimate (default: 4)"
    )
    parser.add_argument(
        "--max-pages", type=int, default=8, help="warn above this estimate (default: 8)"
    )
    args = parser.parse_args(argv)

    source = Path(args.source)
    if not source.exists():
        print(f"error: no such file: {source}", file=sys.stderr)
        return 2

    destination = render(source, out=Path(args.out) if args.out else None)
    pages = estimate_pages(source)

    print(f"wrote {destination}")
    print(f"estimated length: ~{pages} page(s)")
    if pages < args.min_pages:
        print(f"  warning: the brief asks for at least {args.min_pages} pages")
    elif pages > args.max_pages:
        print(f"  warning: the brief caps this at {args.max_pages} pages")
    print("open it and press Ctrl+P -> Save as PDF (margins: Default, background graphics: on)")

    if args.open:
        webbrowser.open(destination.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
