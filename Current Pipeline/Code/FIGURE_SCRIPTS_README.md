These are the Paradigm 2 figure scripts set to Paradigm 1 settings (speeds 0/75/150 deg/s, BlockType "I").
Put them in one folder with Paradigm 1's `pooled_data.csv` and its fit tables (`DDM_*_fits.csv`, `Bayesian_*.csv`), then run each script.
`Bayesian_SRT_ndt.py --replot --mu 40` redraws the forest plot from the saved table without refitting; 40 ms is the population mean printed by the Paradigm 1 fit.
`HRT_floor_control.py` needs `SRT_identifiability.csv`, so run `SRT_identifiability_check.py` first.
