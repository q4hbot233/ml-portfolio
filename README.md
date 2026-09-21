# Seven projects, each with a demo you can run

A small site hosting interactive demos for seven machine learning and data science
projects. Everything runs client-side, with nothing uploaded and no
server involved, but in two different ways.

**Four fit their models in the browser.** These use
[stlite](https://github.com/whitphx/stlite) to run Streamlit on Pyodide/WebAssembly, so the
Python and the model are genuinely executing on the reader's machine:

| Demo | What it asks |
|---|---|
| [A shuttle assistant that refuses](apps/shuttle-assistant/) | Intent routing for a passenger shuttle, judged on how it refuses the questions it cannot answer |
| [Who pays for the model's mistakes?](apps/credit-default/) | Credit default prediction taken past the ROC curve, to the decision and who absorbs its errors |
| [What 492 frauds can and cannot tell you](apps/card-fraud/) | What rebalancing does across 456 configurations, and the alert queue a fraud team works from |
| [The variable that was left out](apps/hmda-denial/) | Mortgage denial where the label is a human decision and the deciding variable is absent by regulation |

**Three render from a pre-computed bundle.** Their inputs (raw telemetry, broadcast audio)
carry licences that do not permit redistribution, so the export ships derived results and the
interaction is over those. No Python runs in the page:

| Demo | What it asks |
|---|---|
| [Race Control](apps/f1-dashboard/) | Five grands prix replayed lap by lap, with six models scoring alongside |
| [Who is driving?](apps/f1-telemetry/) | Whether corner telemetry identifies the driver once the car is held constant |
| [The race you heard](apps/f1-radio/) | Whether team radio sentiment tracks how fans rated the race, and whether the score measures anything |

## What this repository contains

Only **derived results**: summary tables, model outputs and scored predictions.
Where a dataset's licence does not permit redistribution, the underlying data stays out and the
demo works from aggregates instead. The full analyses, with code, tests and executed
notebooks, live in separate private repositories.

## Running it locally

Any static file server will do:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000>.

## Data sources

| Project | Data | Terms |
|---|---|---|
| Credit default | UCI Default of Credit Card Clients (id 350), Yeh 2009 | CC BY 4.0 |
| Card fraud | ULB credit card transactions via OpenML 1597; Dal Pozzolo et al. 2015 | cite the paper |
| Mortgage denial | CFPB HMDA data browser, Georgia 2023 | public disclosure under Regulation C |
| Shuttle assistant | CLINC150, Larson et al. 2019; Avon Transit GTFS feed | CC BY 3.0; CC BY 4.0 |
| Race Control, driver fingerprinting | Timing and telemetry via [FastF1](https://github.com/theOehrly/Fast-F1) | MIT; derived aggregates only |
| Team radio | F1 live-timing radio manifests, transcribed locally with faster-whisper; RaceFans reader ratings | audio and transcripts not redistributed |
