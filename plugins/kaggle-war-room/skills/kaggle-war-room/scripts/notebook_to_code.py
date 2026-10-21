#!/usr/bin/env python3
"""Convert Jupyter notebooks (.ipynb) to plain code without outputs.

- Keeps only code cells (markdown optional with --keep-markdown, as comments).
- Drops all outputs, execution counts and metadata - smaller, diff-friendly, cheaper to read.
- Shell / magic lines (`!pip ...`, `%cd ...`) are kept but turned into comments prefixed
  with `# [SHELL]` / `# [MAGIC]` so reviewers see them and the output still parses as Python.
- Plain .py / .r files are passed through unchanged.

The notebook is only parsed as JSON; nothing in it is executed.

Examples:
  python notebook_to_code.py nb.ipynb                 # print to stdout
  python notebook_to_code.py nb.ipynb -o nb.py
  python notebook_to_code.py a.ipynb b.ipynb --outdir code/
"""
import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass


def _src(cell):
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else str(s)


def convert_line(line):
    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]
    if stripped.startswith("!"):
        return f"{indent}# [SHELL] {stripped}"
    if stripped.startswith("%"):
        return f"{indent}# [MAGIC] {stripped}"
    return line


def notebook_to_code(text, keep_markdown=False):
    nb = json.loads(text)
    cells = nb.get("cells")
    if cells is None:  # nbformat 3
        cells = [c for ws in nb.get("worksheets", []) for c in ws.get("cells", [])]
    out = []
    n = 0
    for cell in cells:
        ctype = cell.get("cell_type")
        if ctype == "code":
            n += 1
            src = _src(cell) if "source" in cell else "".join(cell.get("input", []))
            lines = [convert_line(l) for l in src.splitlines()]
            out.append(f"# %% [cell {n}]")
            out.extend(lines)
            out.append("")
        elif ctype == "markdown" and keep_markdown:
            out.append("# %% [markdown]")
            out.extend("# " + l for l in _src(cell).splitlines())
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def convert_file(path, keep_markdown=False):
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".ipynb":
        return notebook_to_code(text, keep_markdown)
    return text


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help=".ipynb (or .py) files")
    p.add_argument("-o", "--output", help="output file (single input only)")
    p.add_argument("--outdir", help="write <name>.py for each input into this directory")
    p.add_argument("--keep-markdown", action="store_true", help="keep markdown cells as comments")
    args = p.parse_args(argv)

    if args.output and len(args.inputs) > 1:
        p.error("-o works with a single input; use --outdir for several")
    for inp in args.inputs:
        code = convert_file(inp, args.keep_markdown)
        if args.output:
            Path(args.output).write_text(code, encoding="utf-8")
            print(f"{inp} -> {args.output} ({len(code.splitlines())} lines)")
        elif args.outdir:
            out = Path(args.outdir)
            out.mkdir(parents=True, exist_ok=True)
            dest = out / (Path(inp).stem + ".py")
            dest.write_text(code, encoding="utf-8")
            print(f"{inp} -> {dest} ({len(code.splitlines())} lines)")
        else:
            sys.stdout.write(code)
    return 0


if __name__ == "__main__":
    sys.exit(main())
