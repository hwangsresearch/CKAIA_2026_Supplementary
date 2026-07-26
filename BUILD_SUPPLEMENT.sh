#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary_results.tex
latexmk -c supplementary_results.tex
printf 'Built: %s\n' "$ROOT/supplementary_results.pdf"
