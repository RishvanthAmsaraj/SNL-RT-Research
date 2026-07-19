# KINARM reaction-time analysis app

A point-and-click app that reproduces the SNL-RT-Research reaction-time pipeline.
Upload a trial file, map the columns, choose a fitting mode, and download a report
with every table and figure. No IDE, no editing scripts.

The interface is a modern, themed single page — a guided four-step flow (Load →
Filter → Fit → Results) with an animated progress indicator, card-based sections,
and smooth transitions. Theme and layout live in `.streamlit/config.toml` and
`kinarm_rt/ui.py`.

It fits the same models as the repository:

- a **single-boundary shifted-Wald** (drift *v*, boundary *a*, non-decision time
  *t₀*), estimated hierarchically with partial pooling across participant × speed
  units, using the exact likelihood, priors, and bounds from `CODE_REFERENCE.md`;
- an **express/regular two-component Wald mixture** for bimodal saccade cells,
  flagged by Hartigan's dip test (with a Gaussian-mixture fallback);
- the **LATER reciprobit model** for saccades (no non-decision parameter, so
  nothing can floor).

It also adds analyses beyond the basic fit:

- a **non-decision-time dissociation** battery (Friedman, participant bootstrap,
  and permutation tests) for the hand-vs-eye speed effect;
- **fixed-t₀ sensitivity** and an **identifiability sweep** for saccades;
- **mixture-threshold sensitivity** and model-free **vincentiles**;
- a **parameter-recovery study** that shows hand *t₀* is recovered while saccadic
  *t₀* is not;
- **model comparison** by PSIS-LOO (estimated vs fixed *t₀*) and a **frequentist
  Method A** fit (differential evolution) for a Method-A-vs-B check;
- a **per-speed hierarchical model** (group parameters per speed with credible
  intervals) and an optional **LKJ correlated-effects** version that reveals how
  participants' drift, boundary, and non-decision time covary;
- **repo-format CSV export** (`Bayesian_hrt_fits.csv`, `Bayesian_srt_fits.csv`)
  that drops straight into the pipeline's downstream scripts.

The core fit is **validated against the real `pooled_data.csv`**: it reproduces
`Bayesian_hrt_fits.csv` with per-cell t₀ correlation r = 0.999 (mean difference
0.4 ms) and LATER median r² = 0.971, matching the published values.

Hand *t₀* is identified above the 130 ms floor; saccadic *t₀* floors at 70 ms and
is reported as fixed — the app reproduces that diagnosis rather than hiding it.

---

## Run it — three ways

You only need one. **Docker is the most robust and is identical on macOS and
Windows.** Conda is best if you already use it. Pip works but PyMC can be fiddly
to build on Windows.

### Option A — Docker (most robust, same on every OS)

Install Docker Desktop, then from this folder:

```bash
docker build -t kinarm-rt .
docker run -p 8501:8501 kinarm-rt
```

Open <http://localhost:8501>. This uses the conda-forge PyMC build, so there is
no compiler setup on any operating system.

### Option B — conda (recommended for local use)

```bash
conda env create -f environment.yml
conda activate kinarm-rt
streamlit run app.py
```

### Option C — pip

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

If PyMC fails to build under pip on Windows, use Option A or B — both avoid the
C-compiler step. The app still runs the **fast preview** and the **LATER model**
even if PyMC is missing; only the full Bayesian fit needs it.

### One-click launchers

- macOS / Linux: `./run_app.sh` (run `chmod +x run_app.sh` once)
- Windows: double-click `run_app.bat`

They start the app if it is set up, and print setup instructions if it is not.

---

## Using the app

Four steps on the page: **Load → Filter → Fit → Results**.

- **Load.** Upload a trial file or load the built-in example. The repository's
  wide `pooled_data.csv` works directly — map the hand and saccade RT columns,
  the speed column, and set the BlockType filter to `I`.
- **Filter.** Inclusion windows default to the physiology (hand 150–800 ms,
  saccades 80–600 ms) and are editable.
- **Fit.** Pick effectors and a mode. The **preview** (maximum likelihood)
  returns in seconds. The **full Bayesian** fit runs NUTS and takes a few minutes
  for a full dataset; the page updates when it finishes.
- **Results.** Parameter tables (including express/regular mixture cells),
  convergence and KS diagnostics, figures in the house style, and a one-click
  report download (HTML plus a ZIP of figures and CSVs).

**Timing.** Use the "Fast" sampler preset while setting up, then "Standard" or
"Thorough" (1500/1500/4, matching the repo) for numbers you will report.

The **Advanced analyses** and **Model comparison** tabs hold the dissociation
tests, sensitivity analyses, vincentiles, parameter recovery, LOO comparison, and
the frequentist Method A fit. The fast analyses run on demand from the filtered
data; the comparison and Method A fits refit models, so they take a little time.

---

## Run the whole thing from the command line (no GUI)

For batch or cluster use:

```bash
python run_pipeline.py                          # example data, defaults
python run_pipeline.py --data pooled_data.csv   # your data
python run_pipeline.py --config config.example.yaml
python run_pipeline.py --preview                # fast MLE, no NUTS
```

It writes repo-format CSVs, all figures, the analysis tables, and an HTML report
to the output folder. Edit `config.example.yaml` to control the mapping, windows,
sampler settings, and which analyses to run.

---

## Give it to someone who will not install anything

- **Docker** (above) — hand them the folder and two commands.
- **Streamlit Community Cloud** — push to a GitHub repo and deploy for free at
  share.streamlit.io; users get a URL. The free tier is slow for NUTS, so the
  preview mode is the better default there.

---

## Data format

The app accepts the repository's **wide** layout (one row per trial with both
RTs) or a **long** layout (one RT column plus an effector column). Columns can be
named anything; you map them in the app. Recognised repository columns:

| meaning            | repository column            |
|--------------------|------------------------------|
| participant id     | `Participant`                |
| hand RT (ms)       | `HandRT_ms`                  |
| saccade RT (ms)    | `GazeSRT_ms`                 |
| speed              | `Speed_deg_per_s` or `SpeedCode` (1/2/3) |
| interception trials| `BlockType` == `I`           |

RT units are auto-detected (ms vs s) or you can set them explicitly.

---

## What runs without PyMC

| Feature                          | Needs PyMC? |
|----------------------------------|-------------|
| Loading, filtering, data checks  | no          |
| MLE preview (v, a, t₀)           | no          |
| LATER reciprobit + figures       | no          |
| Report / bundle export           | no          |
| Full hierarchical Bayesian fit   | **yes**     |
| Express/regular Bayesian mixture | **yes**     |

`diptest` is recommended (it is the repository's bimodality test); without it the
app falls back to a Gaussian-mixture BIC comparison.

---

## Project layout

```
app.py                     the Streamlit GUI
kinarm_rt/
  _speeds.py               constants (bounds, filters, floors) from CODE_REFERENCE.md
  data.py                  loading (wide/long), validation, synthetic data
  filters.py               physiological inclusion windows
  models/wald.py           shifted-Wald: pooled hierarchical + mixture + MLE preview
  models/later.py          LATER reciprobit model
  diagnostics.py           goodness of fit, convergence summary
  figures.py               publication-style figures (repo palette)
  report.py                HTML / ZIP export
sample_data/               a ready-to-load example (repository's wide shape)
tests/                     smoke tests (pytest)
environment.yml            conda environment
Dockerfile                 reproducible container
RESEARCH_AND_ROADMAP.md    review + improvements aligned to the repo's own roadmap
```
