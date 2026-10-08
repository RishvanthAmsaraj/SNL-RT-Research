"""The interface: every view of every experiment renders, and every navigation control works without an exception."""
import os, time
import pytest, streamlit
from streamlit.testing.v1 import AppTest
from conftest import APP

APP_PY = os.path.join(APP, "app.py")
# AppTest before Streamlit 1.55 cannot serialise single-choice segmented controls when a button is clicked (it iterates
# the value string). Rendering is still tested there; clicks are covered by the browser check in desktop/ui_check.py.
_V = tuple(int(x) for x in streamlit.__version__.split(".")[:2])
interactive = pytest.mark.skipif(_V < (1, 55), reason="AppTest click support for segmented controls needs Streamlit >= 1.55")


def _app(qp=None, **state):
    at = AppTest.from_file(APP_PY, default_timeout=120)
    for k, v in (qp or {}).items(): at.query_params[k] = v
    for k, v in state.items(): at.session_state[k] = v
    at.run(); return _ok(at)


def _ok(at):
    assert not at.exception, at.exception
    bad = [m.value for m in at.markdown if "could not be drawn" in m.value]
    assert not bad, bad
    return at


def _text(at): return " ".join(m.value for m in at.markdown)


def test_every_view_renders():
    for exp in ["Experiment 1", "Experiment 2"]:
        for view in ["Overview", "Run", "Figures", "Tables", "Documents", "Models"]:
            _app(exp_w=exp, exp_last=exp, view_w=view, view_last=view)
    for cv in ["Side by side", "Figures", "Tables", "Documents"]:
        _app(exp_w="Compare", exp_last="Compare", cview_w=cv, cview_last=cv)


def test_gallery_shows_every_figure():
    from kinarm_rt.engine import results
    for exp, key in (("Experiment 1", "E1"), ("Experiment 2", "E2")):
        rs = results.reference(key); n, files = len(rs.image_stems()), results.original_figure_files(rs)
        at = _app(exp_w=exp, exp_last=exp, view_w="Figures", view_last="Figures")
        assert f"{n} figures in these results ({files} files, PNG and PDF), {n} shown" in _text(at), (exp, n, files)
    assert results.original_figure_files(results.reference("E1")) >= 100 and len(results.reference("E2").image_stems()) == 30


def test_every_table_listed():
    from kinarm_rt.engine import results
    for exp, key in (("Experiment 1", "E1"), ("Experiment 2", "E2")):
        n = len(results.reference(key).tables())
        at = _app(exp_w=exp, exp_last=exp, view_w="Tables", view_last="Tables")
        assert f"{n} tables in these results" in _text(at), (exp, n)


def test_documents_render():
    for exp in ("Experiment 1", "Experiment 2"):
        at = _app(exp_w=exp, exp_last=exp, view_w="Documents", view_last="Documents")
        assert "Documents" in _text(at)
    at = _app(qp={"exp": "compare", "cview": "Documents"}); assert "Documents" in _text(at)


def test_deep_link():
    at = _app(qp={"exp": "E2", "view": "Figures", "fig": "Bayesian_srt_ndt"})
    assert "Saccadic t₀ per participant" in _text(at) and "What it shows" in _text(at)


@interactive
def test_overview_and_figure_navigation():
    at = _app()
    at.button(key="go_fig_E1").click().run(); _ok(at); assert at.session_state["view_w"] == "Figures"
    at.button(key="open_E1_NDT_barchart_bayesian").click().run(); _ok(at)
    assert at.session_state["fig"] == "NDT_barchart_bayesian" and "What it shows" in _text(at)
    at.button(key="next_E1").click().run(); _ok(at); assert at.session_state["fig"] != "NDT_barchart_bayesian"
    at.button(key="prev_E1").click().run(); _ok(at); assert at.session_state["fig"] == "NDT_barchart_bayesian"
    at.button(key="close_E1").click().run(); _ok(at); assert at.session_state["fig"] is None
    at.button(key="cmpb_E1_NDT_barchart_bayesian").click().run(); _ok(at)
    assert at.session_state["exp_w"] == "Compare" and at.session_state["sbs"]["L"] == ("E1:ref", "NDT_barchart_bayesian")


@interactive
def test_search_filters():
    at = _app(exp_w="Experiment 2", exp_last="Experiment 2", view_w="Figures", view_last="Figures")
    at.text_input(key="q_E2").input("schematic").run(); _ok(at)
    assert "16 shown" in _text(at)


@interactive
def test_side_by_side():
    at = _app(exp_w="Compare", exp_last="Compare", cview_w="Side by side", cview_last="Side by side")
    for stem in ["NDT_barchart_bayesian", "LATER_reciprobit", "HRT_floor_control", "vincentile_results_fig3_vincentile_by_speed"]:
        at.button(key=f"preset_{stem}").click().run(); _ok(at)
        assert at.session_state["sbs"]["L"] == ("E1:ref", stem) and at.session_state["sbs"]["R"] == ("E2:ref", stem)
    at.selectbox(key="sbs_src_R").select("Comparison figures").run(); _ok(at)
    at.selectbox(key="sbs_fig_R").select("Where the hand speed effect is").run(); _ok(at)
    assert at.session_state["sbs"]["R"] == ("CMP:ref", "P1_vs_P2_hand")


@interactive
def test_models_and_two_boundary_links():
    at = _app(exp_w="Experiment 1", exp_last="Experiment 1", view_w="Models", view_last="Models")
    assert "Why it is deprecated" in _text(at)
    at.button(key="mfig_E1_two_boundary").click().run(); _ok(at)
    assert at.session_state["view_w"] == "Figures" and at.session_state["cats_E1"] == ["Deprecated"]
    at = _app(exp_w="Experiment 1", exp_last="Experiment 1", view_w="Run", view_last="Run")
    at.button(key="see_E1_two_boundary").click().run(); _ok(at); assert at.session_state["view_w"] == "Figures"


@interactive
def test_run_flow_with_sample_data(tmp_path, monkeypatch):
    monkeypatch.setenv("KINARM_RT_RUNS", str(tmp_path))
    at = _app(exp_w="Experiment 1", exp_last="Experiment 1", view_w="Run", view_last="Run")
    at.toggle(key="sample_E1").set_value(True).run(); _ok(at)
    assert "Data checked" in _text(at)
    for a in ("bayesian", "identifiability"): at.toggle(key=f"tg_E1_{a}").set_value(False).run(); _ok(at)
    at.button(key="go_E1").click().run(); _ok(at)
    run_dir = at.session_state["runs"]["E1"]
    from kinarm_rt.engine import runner
    for _ in range(150):
        if not runner.is_running(run_dir): break
        time.sleep(1)
    at.run(); _ok(at)
    assert "Run finished" in _text(at)
    at.button(key="after_fig_E1").click().run(); _ok(at)
    assert at.session_state["view_w"] == "Figures" and at.session_state["source"]["E1"] == "Your run"


def test_figure_groups_are_main_diagnostic_method_a_deprecated():
    for exp in ("Experiment 1", "Experiment 2"):
        at = _app(exp_w=exp, exp_last=exp, view_w="Figures", view_last="Figures")
        chips = [m.value for m in at.markdown if m.value.startswith('<div class="k-group-chips">')][0]
        assert "Supporting" not in chips and "Earlier versions" not in chips and "Main<b>" in chips and "Deprecated<b>" in chips, chips


def test_figure_page_has_the_full_explanation():
    at = _app(qp={"exp": "E1", "view": "Figures", "fig": "NDT_barchart_bayesian"})
    t = _text(at)
    for part in ("What it shows", "Why it is here", "How to read it", "In these results", "What it means for the project", "170 / 158 / 148"):
        assert part in t, part


def test_side_by_side_explains_the_comparison():
    at = _app(exp_w="Compare", exp_last="Compare", cview_w="Side by side", cview_last="Side by side")
    t = _text(at)
    assert "What this comparison shows" in t and "158 vs 147" in t, t[-800:]
