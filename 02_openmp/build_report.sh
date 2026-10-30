#!/bin/sh
set -e
cd "$(dirname "$0")"
make solver
mkdir -p dumps results/history
for g in "20 20" "40 40" "80 80" "800 1200"; do
    set -- $g
    [ -f dumps/sol_$1x$2.bin ] || OMP_NUM_THREADS=4 ./solver $1 $2 --dump dumps/sol_$1x$2.bin
done
for g in "80 80" "400 600" "800 1200"; do
    set -- $g
    [ -f dumps/ref_$1x$2.bin ] || OMP_NUM_THREADS=4 ./solver $1 $2 1e-12 --dump dumps/ref_$1x$2.bin
    [ -f results/history/real_$1x$2.txt ] || OMP_NUM_THREADS=4 ./solver $1 $2 --history results/history/real_$1x$2.txt --ref dumps/ref_$1x$2.bin
    [ -f results/history/exact_$1x$2.txt ] || OMP_NUM_THREADS=4 ./solver $1 $2 1e-10 --exact --history results/history/exact_$1x$2.txt
done
python3 make_table.py runs/table1/table1.*.out > /dev/null
python3 make_figures.py dumps/sol_20x20.bin dumps/sol_40x40.bin dumps/sol_80x80.bin dumps/sol_800x1200.bin
cd report
xelatex -interaction=nonstopmode -halt-on-error report.tex > build.log
xelatex -interaction=nonstopmode -halt-on-error report.tex > build.log
echo "report: $(pwd)/report.pdf"
