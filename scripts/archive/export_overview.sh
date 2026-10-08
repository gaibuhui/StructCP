#!/usr/bin/env bash
# Export the draw.io source (docs/figs/overview.drawio) to docs/figs/overview.pdf
# for inclusion in the paper. Run on a machine that has draw.io (Electron CLI) or
# the `drawio-export` npm package installed. LibreOffice alone CANNOT open .drawio.
#
# Option A -- official draw.io desktop CLI (https://github.com/jgraph/drawio):
#   drawio --export --format pdf --scale 2 docs/figs/overview.drawio \
#          -o docs/figs/overview.pdf
#
# Option B -- drawio-export (npm, headless chromium under the hood):
#   npx drawio-export docs/figs/overview.drawio --format pdf --scale 2 \
#       -o docs/figs/overview.pdf
#
# After export, just recompile the paper; paper_iclr.tex already includes
# \includegraphics[width=0.95\linewidth]{figs/overview.pdf}.
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$HERE/docs/figs/overview.drawio"
OUT="$HERE/docs/figs/overview.pdf"

if command -v drawio >/dev/null 2>&1; then
  drawio --export --format pdf --scale 2 "$SRC" -o "$OUT"
elif command -v npx >/dev/null 2>&1; then
  npx drawio-export "$SRC" --format pdf --scale 2 -o "$OUT"
else
  echo "ERROR: neither 'drawio' CLI nor 'npx' found." >&2
  echo "Open docs/figs/overview.drawio in draw.io / VS Code (Draw.io Integration)," >&2
  echo "then File -> Export As -> PDF and overwrite docs/figs/overview.pdf." >&2
  exit 1
fi
echo "wrote $OUT"
