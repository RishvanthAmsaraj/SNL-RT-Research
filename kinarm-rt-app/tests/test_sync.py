"""The vendored scripts are the repository's scripts, byte for byte."""
import json, os, subprocess, sys
from conftest import APP, REPO
from kinarm_rt.engine.runner import file_sha


def test_vendored_scripts_match_manifest():
    man = json.load(open(os.path.join(APP, "pipelines", "MANIFEST.json")))
    n = 0
    for name, files in man["pipelines"].items():
        for rec in files:
            assert file_sha(os.path.join(APP, "pipelines", name, rec["file"])) == rec["sha256"], rec["file"]; n += 1
    assert n >= 35


def test_vendored_scripts_match_repository():
    if not os.path.isdir(os.path.join(REPO, "Current Pipeline")):
        return                                        # app copied out of the repo: the manifest test above still holds
    r = subprocess.run([sys.executable, os.path.join(APP, "tools", "sync_pipelines.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout
