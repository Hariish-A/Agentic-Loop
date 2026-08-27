"""Render a Markdown document to a Word-openable ``.doc`` file.

    python scripts/make_doc.py docs/reference.md
    # writes reference.doc next to it; double-click to open in Word

Word has read HTML saved with a ``.doc`` extension since Office 2000, and it
keeps headings, tables, lists and code blocks intact and editable. That is worth
more here than a "real" binary ``.doc``: python-docx cannot write the legacy
binary format at all, and the toolchains that can (LibreOffice, a GTK stack for
WeasyPrint) are a heavy dependency to add for one deliverable.

The stylesheet is print-oriented rather than screen-oriented, and monochrome
throughout, because the first thing anyone does with a .doc is print it or
export it to PDF.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Word maps CSS conservatively, so everything here is deliberately basic:
# point sizes rather than rem, no flexbox, no grid. Black text throughout --
# a document that prints identically in colour and greyscale is one fewer thing
# to check before handing it over.
STYLESHEET = """
  @page { size: A4; margin: 2.2cm; }
  body { font-family: Calibri, Arial, sans-serif; font-size: 11pt;
         line-height: 1.45; color: #000000; }
  h1 { font-size: 17pt; margin-top: 0; margin-bottom: 14pt; color: #000000; }
  h2 { font-size: 14pt; margin-top: 20pt; margin-bottom: 6pt; color: #000000;
       page-break-before: always; }
  h2:first-of-type { page-break-before: avoid; }
  h3 { font-size: 12pt; margin-top: 14pt; margin-bottom: 4pt; color: #000000; }
  h4 { font-size: 11pt; margin-top: 11pt; color: #000000; }
  h1, h2, h3, h4 { page-break-after: avoid; }
  p, li { orphans: 2; widows: 2; color: #000000; }
  ul { margin-top: 4pt; margin-bottom: 8pt; }
  li { margin-bottom: 4pt; }
  strong { font-weight: bold; }
  code { font-family: Consolas, 'Courier New', monospace; font-size: 10pt;
         color: #000000; }
  pre { font-family: Consolas, 'Courier New', monospace; font-size: 9.5pt;
        color: #000000; line-height: 1.3; page-break-inside: avoid;
        white-space: pre-wrap; margin-left: 12pt; }
  blockquote { margin-left: 18pt; margin-right: 18pt; font-style: italic;
               color: #000000; }
  hr { border: none; border-top: 1px solid #000000; margin: 14pt 0; }
  a { color: #000000; text-decoration: none; }
"""

TEMPLATE = """<html xmlns:o="urn:schemas-microsoft-com:office:office"
      xmlns:w="urn:schemas-microsoft-com:office:word"
      xmlns="http://www.w3.org/TR/REC-html40">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<meta charset="utf-8">
<meta name="ProgId" content="Word.Document">
<meta name="Generator" content="Microsoft Word">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""


def render(source: Path, out: Path | None = None) -> Path:
    """Convert one Markdown file to a Word-openable .doc next to it."""
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
    destination = out or source.with_suffix(".doc")
    # utf-8-sig, not utf-8: Word decides a stray .doc file's encoding by
    # sniffing, and without a BOM it falls back to the system code page --
    # which turns every em dash and arrow in this document into mojibake.
    destination.write_text(
        TEMPLATE.format(title=title, css=STYLESHEET, body=body), encoding="utf-8-sig"
    )
    return destination


def estimate_pages(source: Path) -> int:
    """Rough page count: prose at ~520 words a page, tables and code cost more."""
    text = source.read_text(encoding="utf-8")
    words = len(re.findall(r"\S+", text))
    table_rows = len(re.findall(r"^\s*\|", text, re.MULTILINE))
    code_lines = sum(
        1 for line in text.splitlines() if line.startswith("    ") or line.startswith("```")
    )
    return max(1, round(words / 520 + table_rows / 34 + code_lines / 46))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render Markdown to a Word-openable .doc")
    parser.add_argument("source", help="the Markdown file to convert")
    parser.add_argument("-o", "--out", help="output path (default: alongside the source)")
    args = parser.parse_args(argv)

    source = Path(args.source)
    if not source.exists():
        print(f"error: {source} not found", file=sys.stderr)
        return 2

    destination = render(source, Path(args.out) if args.out else None)
    print(f"wrote {destination}")
    print(f"estimated length: ~{estimate_pages(source)} page(s)")
    print("open it in Word; File -> Save As -> .docx if you want the native format")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
