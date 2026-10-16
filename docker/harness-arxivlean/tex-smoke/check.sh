#!/bin/sh
set -eu

tex_smoke_dir="$(mktemp -d "${TMPDIR:-/tmp}/matharena-tex.XXXXXX")"
trap 'rm -rf "$tex_smoke_dir"' EXIT HUP INT TERM
for engine in pdf lualatex xelatex; do
    mkdir "$tex_smoke_dir/$engine"
    cp /opt/harness-tex-smoke/*.tex /opt/harness-tex-smoke/*.bib "$tex_smoke_dir/$engine/"
    (
        cd "$tex_smoke_dir/$engine"
        latexmk "-$engine" -interaction=nonstopmode -halt-on-error -file-line-error \
            -latexoption=-no-shell-escape main.tex > compile.log 2>&1 || {
                cat compile.log
                exit 1
            }
        test -s main.pdf
        test -s main.bbl
        if grep -E 'LaTeX Warning: (Citation|Reference).*undefined|There were undefined' main.log; then
            cat compile.log
            exit 1
        fi
        printf 'TeX compilation passed: %s (math, TikZ, local inputs, BibTeX)\n' "$engine"
    )
done
