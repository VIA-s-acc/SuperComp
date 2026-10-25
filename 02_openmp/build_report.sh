#!/bin/sh
set -e
cd "$(dirname "$0")"
make
mkdir -p dumps
for g in "20 20" "40 40" "80 80" "800 1200"; do
    set -- $g
    [ -f dumps/sol_$1x$2.bin ] || OMP_NUM_THREADS=4 ./solver $1 $2 1e-6 dumps/sol_$1x$2.bin
done
python3 make_table.py runs/table1/table1.*.out > /dev/null
python3 make_figures.py dumps/sol_20x20.bin dumps/sol_40x40.bin dumps/sol_80x80.bin dumps/sol_800x1200.bin
cd report
xelatex -interaction=nonstopmode -halt-on-error report.tex > build.log
xelatex -interaction=nonstopmode -halt-on-error report.tex > build.log
echo "report: $(pwd)/report.pdf"
