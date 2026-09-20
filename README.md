# Six projects, each one you can actually run

A small site hosting interactive demos for six machine learning, data science and
quantitative research projects. Every demo runs **entirely in the browser** — real Python,
real models, real data — using [stlite](https://github.com/whitphx/stlite), which runs
Streamlit on Pyodide/WebAssembly. Nothing is uploaded and no server is involved.

| Demo | What it asks |
|---|---|
| [What 492 frauds can tell you](apps/card-fraud/) | Extreme class imbalance, tested rather than assumed, and the queue metric a fraud team actually lives |
| [The variable that was left out](apps/hmda-denial/) | Mortgage denial where the label is a human decision and the deciding variable is absent by regulation |
| [Who pays for the model's mistakes?](apps/credit-default/) | Credit default prediction taken past the ROC curve, to the decision and who absorbs its errors |
| [Is it the car or the driver?](apps/f1-telemetry/) | Diagnosing a suspected car fault from qualifying telemetry, against two reference frames |

## What this repository contains

Only **derived results**: summary tables, model outputs, return series and scored predictions.
Where a dataset's licence does not permit redistribution, the underlying data stays out and the
demo works from aggregates instead. The full analyses — code, tests and executed notebooks —
live in separate private repositories.

## Running it locally

Any static file server will do:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000>.

## Data sources

- Credit default and cloud datasets — UCI Machine Learning Repository, CC BY 4.0
- Card transactions — ULB Machine Learning Group via OpenML 1597; cite Dal Pozzolo et al. (2015)
- Mortgage applications — CFPB HMDA data browser, public disclosure under Regulation C
- Factor and industry returns — Kenneth R. French data library
- MovieLens — GroupLens (aggregates only; the dataset itself is not redistributed)
- Formula 1 telemetry — [FastF1](https://github.com/theOehrly/Fast-F1) (per-corner aggregates only)
- Circuit geometry — [bacinger/f1-circuits](https://github.com/bacinger/f1-circuits)
