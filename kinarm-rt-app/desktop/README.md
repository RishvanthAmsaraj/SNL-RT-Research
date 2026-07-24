# Desktop apps

This folder turns the app into something you double-click: no Python to install, no
terminal, no `streamlit run`. It starts the server on a private local port, opens it
in a window, and shuts everything down when the window closes.

Nothing is left out. Method A, the hierarchical Bayesian fit with NUTS, the LATER
model, every figure and diagnostic, and the full report all work, at the same speed
and with the same numbers as running from the conda environment.

## The one thing to know before you start

**The apps have to be built on the system they run on.** Both PyInstaller and
conda-pack bundle the host machine's Python runtime and compiled libraries, so a
Windows build must be made on Windows and a macOS build on macOS. There is no
cross-compiling, and anyone who hands you a binary built somewhere else is handing
you something they never ran.

Two ways to get builds without owning both machines:

- **GitHub Actions** (recommended) — `.github/workflows/build-desktop.yml` borrows a
  real Windows machine, an Intel Mac and an Apple Silicon Mac from GitHub's hosted
  runners. Free for public repositories. Actions tab → *Build desktop apps* →
  *Run workflow*, wait about half an hour, download the artifacts.
- **Build locally** on each machine you have, with the commands below.

## Why the app is about a gigabyte

It ships a complete conda environment, including a C++ compiler, rather than a
frozen executable.

PyMC runs on PyTensor, which generates C++ for the model's log-likelihood and
compiles it against the Python headers the first time it runs. If no compiler is
present it does not fail — it quietly drops to a pure-Python evaluation path. On a
short test fit here that took **248 seconds instead of 27, about nine times
slower**, and the sampler did not land on the same estimates (t₀ 181.25 ms against
181.5 ms), because the two paths take different routes through the same maths.

A frozen executable cannot fix that: PyInstaller bundles compiled libraries but not
a compiler, and not the Python headers PyTensor needs to compile against. So the
toolchain travels with the app. The size buys a build with nothing missing and no
penalty for using it.

If you would rather have a small download and can live without the Bayesian fit,
the ordinary `requirements.txt` install plus `run_app.bat` / `run_app.sh` is a few
hundred megabytes and runs everything else.

## Building locally

From `kinarm-rt-app`, on the machine you are targeting:

```bash
conda env create -f environment-desktop.yml
conda activate kinarm-rt-desktop

# the compiler, which environment.yml cannot express per platform
conda install -c conda-forge m2w64-toolchain libpython   # Windows
conda install -c conda-forge compilers                   # macOS

python -m pytest -q                # the whole suite, Bayesian tests included
conda pack -n kinarm-rt-desktop -o env.tar.gz
```

Then assemble the bundle as the workflow does: unpack `env.tar.gz` into `env/`, run
`conda-unpack` inside it, copy the app into `app/`, and add the launcher — a small
`stub.py` build on Windows, or the `.app` layout on macOS. The workflow is the
reference; it is easier to run it than to reproduce it by hand.

## Running during development

No packaging needed to try the desktop behaviour:

```bash
python desktop/launcher.py
```

It prints whether PyMC found a compiler, so you can see immediately whether the
Bayesian fit will run at full speed.

## Handing the app to someone else

Neither build is code-signed, because signing needs paid developer accounts (the
Apple Developer Program, and an Authenticode certificate for Windows). Without them:

- **Windows** — SmartScreen warns on first launch. *More info* → *Run anyway*. It
  stops asking afterwards.
- **macOS** — Gatekeeper blocks the first launch. **Right-click the app → Open**,
  then confirm; double-clicking will not offer the option. The workflow applies an
  ad-hoc signature so macOS reports an unsigned app rather than a damaged one, which
  is a far less alarming message.

If this is going to more than a handful of people, the certificates are worth the
cost, and the workflow has an obvious place to add them.

Apple Silicon and Intel Macs need separate builds; the workflow produces both.

## First run is slower

The first hierarchical fit compiles the model and caches it under the user's
application data. Later runs reuse the cache. This is PyTensor doing the work that
makes every subsequent fit fast, not something going wrong.

## How the launcher works

Two processes, and the reason is worth recording because it is not obvious.
Streamlit installs signal handlers at startup, which only works on a process's main
thread. On macOS the native window toolkit also insists on the main thread. Both
cannot have it, so the launcher re-runs itself with a hidden flag: the child serves
the app, the parent shows the window and stops the child when the window closes.

If pywebview is missing or a window cannot be created, the app opens in the default
browser instead. That is a working fallback, not an error, and the console says
which happened.

## Files here

| File | What it does |
|---|---|
| `launcher.py` | Entry point. Provisions the compiler, starts the server, opens the window. |
| `stub.py` | The double-clickable front door on Windows; finds the packed environment. |
| `mac_launch.sh` | The equivalent inside `KINARM RT.app/Contents/MacOS/`. |
| `smoke_test.py` | Runs a built bundle, checks it serves, and fails if PyMC is not usable. |
| `icon.ico` / `icon.icns` | Application icons, if present. Builds fine without them. |
