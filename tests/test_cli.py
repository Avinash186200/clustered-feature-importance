import json

from clusterfi import cli
from clusterfi.scale import run_scale


def test_demo_prints_every_method(capsys):
    cli.main(["demo", "--seed", "3", "--noise-frac", "0.25"])
    out = capsys.readouterr().out
    for m in ("MDI", "MDA", "SHAP", "cMDI", "cMDA", "cSHAP"):
        assert f"\n{m} " in out


def test_scale_smoke_reports_identical_results_and_no_timing_checks(tmp_path):
    res = run_scale(workers=(1, 2), n_valid=300, n_repeats=1, passes=1, noise_frac=0.25)
    assert [r["workers"] for r in res["rows"]] == [1, 2]
    assert all(r["identical_to_serial"] for r in res["rows"])
    # timings are printed, never asserted: CI runners are shared and slow
    json.dumps(res)
