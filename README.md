# QB Evaluator: fault-adjusted EPA backtest

Grades NFL QBs using nflverse play-by-play + FTN charting, separating interceptions by fault
(bad throw vs. tipped/receiver-caused, plus "dropped interceptions").

## Files
- `qb_eval.py` - data loading, counterfactual value models, play-level adjustments, QB tables, 1-100 grade
- `backtest.py` - split-half and year-over-year tests, bootstrap CIs, shrinkage
- `mixed.py` - play-level QB + defense + offense effects model (ridge = random effects)
- `qb_backtest.ipynb` - the full analysis (executed); `qb_backtest.html` is a read-only copy
- `make_notebook.py` - regenerates the notebook
- `qb_grades_2026_wk4.csv` - current live board

## Run
    python -m venv .venv
    source .venv/Scripts/activate   # Git Bash on Windows (PowerShell: .venv\Scripts\activate)
    pip install -r requirements.txt
    jupyter nbconvert --execute --to notebook --inplace qb_backtest.ipynb

First run downloads 2022-2026 data into `data/` (cached after that). To refresh for a new week,
call `q.load_season(2026, refresh=True)` or delete `data/*_2026.parquet`.
