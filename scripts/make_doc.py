"""Render a Markdown document to a Word-openable ``.doc`` file.

    python scripts/make_doc.py docs/reference.md
    # writes reference.doc next to it; double-click to open in Word

Word has read HTML saved with a ``.doc`` extension since Office 2000, and it
keeps headings, tables, lists and code blocks intact and editable. That is worth
more here than a "real" binary ``.doc``: python-docx cannot write the legacy
binary format at all, and the toolchains that can (LibreOffice, a GTK stack for
WeasyPrint) are a heavy dependency to add for one deliverable.

The stylesheet is print-oriented rather than screen-oriented -- serif body,
tables with visible rules, page-break control on headings -- because the first
thing anyone does with a .doc is print it or export it to PDF.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Word maps CSS conservatively. Everything here is deliberately basic: point
# sizes rather than rem, explicit table borders rather than border-collapse
# tricks, and no flexbox or grid anywhere.
STYLESHEET = """
  @page { size: A4; margin: 2cm; }
  body { font-family: Georgia, 'Times New Roman', serif; font-size: 10.5pt;
         line-height: 1.45; color: #1a1a1a; }
  h1 { font-size: 20pt; border-bottom: 2px solid #444; padding-bottom: 4pt;
       margin-top: 22pt; page-break-before: always; }
  h1:first-of-type { page-break-before: avoid; }
  h2 { font-size: 15pt; margin-top: 18pt; color: #14304f; }
  h3 { font-size: 12.5pt; margin-top: 14pt; color: #14304f; }
  h4 { font-size: 11pt; margin-top: 12pt; }
  h1, h2, h3, h4 { page-break-after: avoid; font-family: Calibri, Arial, sans-serif; }
  p, li { orphans: 2; widows: 2; }
  code { font-family: Consolas, 'Courier New', monospace; font-size: 9pt;
         background: #f2f2f2; padding: 0 2pt; }
  pre { font-family: Consolas, 'Courier New', monospace; font-size: 8.5pt;
        background: #f6f6f6; border: 1px solid #d8d8d8; padding: 7pt;
        line-height: 1.3; page-break-inside: avoid; white-space: pre-wrap; }
  pre code { background: none; padding: 0; font-size: 8.5pt; }
  table { border-collapse: collapse; width: 100%; margin: 10pt 0;
          font-size: 9.5pt; font-family: Calibri, Arial, sans-serif; }
  th, td { border: 1px solid #b0b0b0; padding: 4pt 6pt; text-align: left;
           vertical-align: top; }
  th { background: #eaeef2; font-weight: bold; }
  blockquote { border-left: 3px solid #b0b0b0; margin-left: 0; padding-left: 10pt;
               color: #444; font-style: italic; }
  hr { border: none; border-top: 1px solid #c8c8c8; margin: 14pt 0; }
  a { color: #14304f; }
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
