"""Registry, figure catalogue and analysis catalogue agree with each other."""
from kinarm_rt.engine import registry as R
from kinarm_rt import explain as X


def test_every_analysis_step_exists_and_plan_orders_dependencies():
    for exp in R.EXPERIMENTS.values():
        steps = R.steps_for(exp)
        for a in R.analyses_for(exp.id):
            for s in a.steps: assert s in steps, (exp.id, a.id, s)
        order = [s.id for s in R.plan(exp, [a.id for a in R.analyses_for(exp.id)])]
        for i, sid in enumerate(order):
            for d in steps[sid].needs: assert d in order[:i], (sid, d)


def test_every_catalogued_figure_is_written_by_a_step_or_shipped():
    for exp in R.EXPERIMENTS.values():
        outs = {o for s in R.steps_for(exp).values() for o in s.outputs}
        for f in X.FIGURES:
            if exp.id not in f.experiments or f.analysis == "two_boundary": continue
            stems = [f.stem.format(s=s) for s in exp.speeds] if f.per_speed else [f.stem]
            for stem in stems: assert f"{stem}.png" in outs or f"{stem}.pdf" in outs, (exp.id, stem)


def test_deprecated_models_explain_themselves():
    for a in R.ANALYSES:
        if a.status == "deprecated":
            note = X.MODEL_NOTES[a.id]; assert note.why and note.evidence
