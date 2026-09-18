# Cloud classification — separability explorer

An in-browser Streamlit app (stlite / Pyodide). Pick a column, move a threshold, and
watch a one-rule classifier score the 1989 UCI *Cloud* dataset. The point it makes:
single raw columns already score 1.000, so the interesting work in that project was
the engineering, not the modelling.

## What ships here

| File | What it is | Size |
|---|---|---|
| `app.py` | The app. Compute layer (plain functions, no Streamlit) above `main()`. | ~24 KB |
| `data/clouds_features.csv` | 2,048 super-pixels × (10 raw columns + 4 engineered features + `class`). | ~221 KB |
| `data/raw-column-separability.csv` | The separability ladder, copied verbatim from the pipeline's committed report. | ~1 KB |

Nothing else. No model pickles, no run artifacts, no coursework material.

## Provenance

`clouds_features.csv` is the raw UCI file parsed by the private pipeline's own
`create_dataset` step (with my corrected column order and row ranges) and enriched by
its own `generate_features` step, exactly as configured in
`config/default-config.yaml`. The column *names* are my correction — the file's 1989
preamble lists them in the wrong order, and 2,045 of 2,048 rows violate
`min ≤ mean ≤ max` under the documented order.

`raw-column-separability.csv` is not recomputed here; it is the executed output of the
pipeline's walkthrough notebook (5-fold stratified CV, depth-1 decision tree, seed 42).

**Dataset licence.** Collard, P. (1989). *Cloud* [Dataset]. UCI Machine Learning
Repository. <https://doi.org/10.24432/C5359Z>. CC BY 4.0 — redistribution permitted
with attribution, which is why the table ships at all.

## Runtime notes

The app imports only `numpy`, `pandas` and `matplotlib` — no scikit-learn. The optimal
single threshold is found exactly with a sorted cumulative-count sweep in `best_stump`,
which keeps the Pyodide payload small and the page fast. The cross-validated figures on
the page come from the shipped ladder, not from refitting in the browser.

## Testing the compute layer

`app.py` imports cleanly in ordinary Python (the Streamlit import is guarded and the UI
only runs under `__main__`), so every function above `main()` can be called directly:

```python
import app
frame = app.load_features()
labels = frame["class"].to_numpy()
app.best_stump(frame["IR_mean"].to_numpy(), labels)["accuracy"]   # 1.0
app.class_overlap(frame["IR_mean"].to_numpy(), labels)["n_rows_inside"]   # 0
```
