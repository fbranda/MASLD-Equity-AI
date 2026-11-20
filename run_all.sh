#!/bin/sh
set -eu
cd "$(dirname "$0")"
python scripts/run_analysis.py --data data --output results --download --bootstrap 450
python scripts/augment_analysis.py --data data --results results --bootstrap 450
python scripts/make_figures.py --data data --results results
python scripts/build_full_paper.py --results results --out BioMedInformatics_MASLD_Reproduced_Manuscript.docx
printf '\nCompleted: results/results.json and BioMedInformatics_MASLD_Reproduced_Manuscript.docx\n'
