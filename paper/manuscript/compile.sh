#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/texlive/2026/bin/x86_64-linux:$PATH"

cd "$(dirname "$0")"

cleanup() {
    rm -f main.aux main.bbl main.blg main.log main.out main.toc \
        main.fdb_latexmk main.fls main.synctex.gz
}
trap cleanup EXIT

pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex

if [ -f main.pdf ]; then
    printf 'SUCCESS: main.pdf generated (%s)\n' "$(du -h main.pdf | cut -f1)"
else
    printf 'FAILED: main.pdf not generated\n' >&2
    exit 1
fi
