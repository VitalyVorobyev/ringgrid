from __future__ import annotations

import base64
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    sys.path.insert(0, str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


srv = _load("review_server", "tools/review/server.py")
summ = _load("review_summarize", "tools/review/summarize.py")

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

TODAY = {
    "detected_markers": [
        {
            "id": 7,
            "confidence": 0.9,
            "center": [10.0, 20.0],
            "grid_coord": [1, -2],
            "board_xy_mm": [8.0, 4.0],
            "ellipse_outer": {"cx": 10.1, "cy": 20.1, "a": 12.0, "b": 10.0, "angle": 0.5},
            "ellipse_inner": {"cx": 10.1, "cy": 20.1, "a": 6.0, "b": 5.0, "angle": 0.5},
            "center_source": "projective_center",
        },
        {"confidence": 0.4, "center": [100.0, 50.0]},
    ],
    "image_size": [720, 540],
    "center_frame": "image",
}

NEXT = {
    "detected_markers": [{"id": 3, "confidence": 0.8, "center": [30.0, 40.0]}],
    "diagnostics": {
        "markers": [
            {
                "ellipse_outer": {"cx": 30.0, "cy": 40.0, "a": 9.0, "b": 8.0, "angle": 0.0},
                "center_source": "photometric",
                "source": "fit_decode",
                "fit": {"n_points_outer": 31, "rms": 0.2},
                "edge_points_outer": [[1, 2], [3, 4]],
            }
        ]
    },
}


class Normalise(unittest.TestCase):
    def test_current_shape(self):
        n = srv.normalise_detection(TODAY, "t", "r")
        self.assertEqual((n["n_markers"], n["n_decoded"]), (2, 1))
        m = n["markers"][0]
        self.assertEqual((m["id"], m["grid_coord"], m["center_source"]), (7, [1, -2], "projective_center"))
        self.assertEqual(m["ellipse_outer"]["a"], 12.0)
        self.assertIsNone(n["markers"][1]["id"])
        self.assertIsNone(n["markers"][1]["ellipse_outer"])
        self.assertEqual(n["image_size"], [720, 540])
        self.assertEqual(n["radius_hint"], 12.0)

    def test_next_release_shape_reads_diagnostics(self):
        m = srv.normalise_detection(NEXT)["markers"][0]
        self.assertEqual(m["ellipse_outer"]["a"], 9.0)
        self.assertEqual(m["center_source"], "photometric")
        self.assertEqual(m["details"]["fit.n_points_outer"], 31)
        self.assertEqual(m["details"]["source"], "fit_decode")
        self.assertNotIn("edge_points_outer", m["details"])

    def test_misaligned_diagnostics_ignored(self):
        data = dict(NEXT, diagnostics={"markers": []})
        n = srv.normalise_detection(data)
        self.assertIsNone(n["markers"][0]["ellipse_outer"])
        self.assertTrue(n["warnings"])

    def test_malformed_never_raises(self):
        bad = [
            None, 5, "x", [], {}, {"detected_markers": "nope"}, {"detected_markers": [1, None, "a"]},
            {"detected_markers": [{"center": ["a", 1]}]},
            {"detected_markers": [{"center": [float("nan"), 1]}]},
            {"detected_markers": [{"center": [1, 2], "id": "7", "confidence": "high", "grid_coord": [1], "ellipse_outer": {"a": -1}}]},
            {"detected_markers": [{"center": [1, 2], "id": True}], "diagnostics": {"markers": "x"}},
            {"detected_markers": [{"center": [1, 2]}], "image_size": [1]},
        ]
        for b in bad:
            n = srv.normalise_detection(b)
            json.dumps(n, allow_nan=False)  # fully JSON-safe
            for m in n["markers"]:
                self.assertEqual(len(m["center"]), 2)

    def test_garbled_fields_become_none(self):
        data = {"detected_markers": [{"center": [1, 2], "id": "7", "confidence": "high", "grid_coord": [1], "ellipse_outer": {"a": -1}}]}
        m = srv.normalise_detection(data)["markers"][0]
        self.assertIsNone(m["id"])
        self.assertIsNone(m["confidence"])
        self.assertIsNone(m["grid_coord"])
        self.assertIsNone(m["ellipse_outer"])

    def test_centre_falls_back_to_ellipse(self):
        data = {"detected_markers": [{"ellipse_outer": {"cx": 5, "cy": 6, "a": 3, "b": 2, "angle": 0}}]}
        self.assertEqual(srv.normalise_detection(data)["markers"][0]["center"], [5.0, 6.0])

    def test_radius_hint_from_neighbours_and_fallback(self):
        pts = {"detected_markers": [{"center": [0, 0]}, {"center": [100, 0]}]}
        self.assertAlmostEqual(srv.normalise_detection(pts)["radius_hint"], 35.0)
        self.assertEqual(srv.normalise_detection({"detected_markers": [{"center": [0, 0]}]})["radius_hint"], srv.DEFAULT_MARKER_RADIUS_PX)


class Compare(unittest.TestCase):
    def test_classification(self):
        def mk(i, id_, x, y):
            return {"index": i, "id": id_, "center": [x, y]}

        a = [mk(0, 1, 0, 0), mk(1, 2, 50, 0), mk(2, 3, 100, 0), mk(3, 4, 150, 0)]
        b = [mk(0, 1, 0.1, 0), mk(1, 9, 50, 0), mk(2, 3, 103, 0), mk(3, 7, 400, 0)]
        c = srv.compare_markers(a, b)
        cls = {p["a"]: p["cls"] for p in c["pairs"]}
        self.assertEqual(cls, {0: "same", 1: "id_differs", 2: "position_differs"})
        self.assertEqual((c["only_a"], c["only_b"]), ([3], [3]))
        self.assertTrue(srv.differs(c["counts"]))
        self.assertFalse(srv.differs(srv.compare_markers(a, a)["counts"]))


class Feedback(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.store = srv.FeedbackStore(self.tmp / "feedback.json", "ds", ["a", "b"])

    @staticmethod
    def tile():
        return {
            "verdict": "has_issues", "remark": "r", "reviewed_at": "2026-01-01T00:00:00+00:00", "runs_seen": ["r1"],
            "marker_flags": [{"run": "r1", "kind": "wrong_id", "marker": {"index": 3, "id": 5, "center": [1.0, 2.0]}, "correct_id": 6, "remark": "x", "at": "t"}],
            "pins": [{"uid": "p1", "type": "missed_marker", "x": 3, "y": 4, "true_id": 9, "remark": "m"}, {"uid": "p2", "type": "note", "x": 1, "y": 1, "text": "hi"}],
        }

    def test_round_trip(self):
        self.store.patch({"a": self.tile()}, [{"at": "t", "text": "hello"}])
        on_disk = json.loads((self.tmp / "feedback.json").read_text())
        self.assertEqual(on_disk["schema"], srv.SCHEMA)
        self.assertEqual(on_disk["tiles"]["a"]["marker_flags"][0]["correct_id"], 6)
        self.assertEqual(on_disk["general_remarks"][0]["text"], "hello")
        again = srv.FeedbackStore(self.tmp / "feedback.json", "ds", ["a", "b"])
        self.assertEqual(again.data["tiles"]["a"], on_disk["tiles"]["a"])

    def test_cleared_tile_is_dropped(self):
        self.store.patch({"a": self.tile()}, None)
        self.store.patch({"a": {"verdict": None, "remark": "", "marker_flags": [], "pins": []}}, None)
        self.assertNotIn("a", self.store.data["tiles"])

    def test_validation_rejects(self):
        t = self.tile()
        cases = [
            lambda t: t.update(verdict="great"),
            lambda t: t.update(extra=1),
            lambda t: t["marker_flags"][0].update(kind="nope"),
            lambda t: t["marker_flags"][0]["marker"].update(center=[1]),
            lambda t: t["marker_flags"][0].update(correct_id=-1),
            lambda t: t["pins"][0].update(type="blob"),
            lambda t: t["pins"][1].update(true_id=3),
            lambda t: t["pins"].append(dict(t["pins"][0])),
            lambda t: t.update(runs_seen=["../x"]),
        ]
        for mutate in cases:
            t = self.tile()
            mutate(t)
            with self.assertRaises(srv.FeedbackError, msg=str(mutate)):
                self.store.patch({"a": t}, None)
        with self.assertRaises(srv.FeedbackError):
            self.store.patch({"zzz": self.tile()}, None)
        with self.assertRaises(srv.FeedbackError):
            self.store.replace({"schema": "other", "dataset": "d", "tiles": {}})
        self.assertFalse((self.tmp / "feedback.json").exists())  # nothing written on rejection

    def test_atomic_write_keeps_old_file_on_failure(self):
        self.store.patch({"a": self.tile()}, None)
        before = (self.tmp / "feedback.json").read_text()
        with mock.patch.object(srv.os, "replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.store.patch({"b": self.tile()}, None)
        self.assertEqual((self.tmp / "feedback.json").read_text(), before)
        self.assertEqual([p.name for p in self.tmp.iterdir()], ["feedback.json"])  # temp file cleaned up
        self.assertNotIn("b", self.store.data["tiles"])

    def test_corrupt_existing_file_not_overwritten(self):
        (self.tmp / "feedback.json").write_text("{not json")
        with self.assertRaises(SystemExit):
            srv.FeedbackStore(self.tmp / "feedback.json", "ds", ["a"])
        self.assertEqual((self.tmp / "feedback.json").read_text(), "{not json")


class Loopback(unittest.TestCase):
    def test_refuses_non_loopback(self):
        for host in ("0.0.0.0", "192.168.1.5", "::", "example.com", ""):
            with self.assertRaises(ValueError, msg=host):
                srv.require_loopback(host)
        for host in ("127.0.0.1", "localhost", "127.0.0.2", "::1"):
            srv.require_loopback(host)


@unittest.skipUnless(shutil.which("git"), "git not available")
class GitIgnore(unittest.TestCase):
    def test_ignored_and_unignored(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        subprocess.run(["git", "init", "-q", str(tmp)], check=True)
        (tmp / ".gitignore").write_text("out/\n")
        self.assertTrue(srv.check_output_ignored(tmp / "out" / "review" / "x")[0])
        ok, msg = srv.check_output_ignored(tmp / "visible")
        self.assertFalse(ok)
        self.assertIn("NOT ignored", msg)

    def test_outside_repo_is_fine(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        if subprocess.run(["git", "-C", str(tmp), "rev-parse"], capture_output=True).returncode == 0:
            self.skipTest("tmp lies inside a git repo")
        self.assertTrue(srv.check_output_ignored(tmp / "o")[0])

    def test_repo_default_output_is_ignored(self):
        ok, msg = srv.check_output_ignored(ROOT / "tools" / "out" / "review" / "_probe_gitignore_check")
        shutil.rmtree(ROOT / "tools" / "out" / "review" / "_probe_gitignore_check", ignore_errors=True)
        self.assertTrue(ok, msg)


FAKE_DETECTOR = """#!/bin/sh
if [ "$1" = "--version" ]; then echo "fake-detector 9.9"; exit 0; fi
while [ $# -gt 0 ]; do if [ "$1" = "--out" ]; then OUT="$2"; fi; shift; done
echo '{"detected_markers":[{"id":1,"confidence":0.5,"center":[10,10]},{"id":2,"center":[50,50]}],"image_size":[64,48]}' > "$OUT"
"""


def _png(path: Path, w=64, h=48):
    Image.new("L", (w, h), 90).save(path)


@unittest.skipIf(Image is None, "Pillow not available")
class Http(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.data = cls.tmp / "data"
        cls.data.mkdir()
        _png(cls.data / "strip_a.png", 128, 48)
        _png(cls.data / "strip_b.png", 128, 48)
        cls.out = cls.tmp / "out"
        cls.out.mkdir()
        cls.tmp = cls.tmp.resolve()
        cls.fake = cls.tmp / "fakedet"
        cls.fake.write_text(FAKE_DETECTOR)
        cls.fake.chmod(cls.fake.stat().st_mode | stat.S_IEXEC)
        ds = srv.Dataset(cls.data, 2, cls.out)
        cls.app = srv.App(ds, cls.out, {"fake": cls.fake.resolve()})
        imported = cls.tmp / "imported"
        imported.mkdir()
        (imported / "strip_a_00_whatever.json").write_text(json.dumps(NEXT))
        cls.app.runs.import_dir("imp", imported)
        cls.server = srv.make_server(cls.app, "127.0.0.1", 0)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def req(self, path, body=None, headers=None):
        url = f"http://127.0.0.1:{self.port}{path}"
        data = None if body is None else json.dumps(body).encode()
        r = urllib.request.Request(url, data=data, headers=headers or {})
        if data is not None:
            r.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(r, timeout=20) as resp:
                return resp.status, resp.read(), resp.headers
        except urllib.error.HTTPError as e:
            return e.code, e.read(), e.headers

    def test_state_index_and_csp(self):
        code, body, hdr = self.req("/")
        self.assertEqual(code, 200)
        self.assertIn("default-src 'none'", hdr["Content-Security-Policy"])
        self.assertNotRegex(body.decode(), r"(src|href)=[\"']https?://")
        code, body, _ = self.req("/api/state")
        st = json.loads(body)
        names = [t["name"] for t in st["dataset"]["tiles"]]
        self.assertEqual(names, ["strip_a_00", "strip_a_01", "strip_b_00", "strip_b_01"])
        self.assertEqual(st["dataset"]["tiles"][0]["width"], 64)
        self.assertEqual(st["feedback"]["schema"], srv.SCHEMA)

    def test_screen_target_endpoint(self):
        try:
            from ringgrid import TargetLayout
        except ImportError:
            self.skipTest("RingGrid bindings are unavailable")
        if not hasattr(TargetLayout, "render_fitted_screen_target"):
            self.skipTest("RingGrid bindings lack screen rendering")
        body = {
            "template": "coded_hex",
            "screen": {
                "width_px": 825, "height_px": 1200, "pixel_pitch_mm": 25.4 / 150,
                "marker_diameter_mm": 10.752,
            },
        }
        code, data, _ = self.req("/api/screen-target", body)
        self.assertEqual(code, 200, data[:300])
        response = json.loads(data)
        self.assertEqual(response["metadata"]["marker_count"], 152)
        self.assertEqual(json.loads(response["json_text"])["lattice"]["rows"], 16)
        self.assertTrue(base64.b64decode(response["png_base64"]).startswith(b"\x89PNG"))
        self.assertEqual(self.req("/api/screen-target", {**body, "template": "../unknown"})[0], 400)
        self.assertEqual(self.req("/api/screen-target", body, {"Origin": "https://example.com"})[0], 403)

    def test_tile_image_is_a_cropped_png(self):
        code, body, hdr = self.req("/api/tile/strip_a_01/image")
        self.assertEqual((code, hdr["Content-Type"]), (200, "image/png"))
        self.assertEqual(srv.png_size(self.out / "tiles" / "strip_a_01.png"), (64, 48))
        self.assertTrue(body.startswith(b"\x89PNG"))

    def test_detection_endpoint_normalised(self):
        code, body, _ = self.req("/api/run/imp/tile/strip_a_00")
        self.assertEqual(code, 200)
        d = json.loads(body)
        self.assertEqual(d["markers"][0]["ellipse_outer"]["a"], 9.0)
        self.assertEqual(self.req("/api/run/imp/tile/strip_b_00")[0], 404)

    def test_path_traversal_rejected(self):
        for p in (
            "/api/tile/..%2f..%2fetc%2fpasswd/image", "/api/tile/../../etc/passwd/image", "/api/tile/%2e%2e/image",
            "/api/run/..%2f/tile/strip_a_00", "/api/run/imp/tile/..%2f..%2fx", "/api/run/imp/tile/strip_a_00%00",
            "/api/compare/..%2fx?a=imp&b=imp", "/..%2f..%2fetc/passwd", "/api/tile/strip_a_00/image/../..",
        ):
            code, _, _ = self.req(p)
            self.assertIn(code, (400, 404), p)
        code, _, _ = self.req("/api/runs", {"name": "../../evil", "binary": str(self.fake), "args": ""})
        self.assertIn(code, (400, 403))
        self.assertFalse((self.tmp / "evil").exists())

    def test_feedback_api_round_trip(self):
        tile = Feedback.tile()
        code, body, _ = self.req("/api/feedback", {"tiles": {"strip_a_00": tile}, "general_remarks": [{"at": "t", "text": "g"}]})
        self.assertEqual(code, 200, body)
        on_disk = json.loads((self.out / "feedback.json").read_text())
        self.assertEqual(on_disk["tiles"]["strip_a_00"]["pins"][1]["text"], "hi")
        self.assertIn("strip_a_00", json.loads(self.req("/api/state")[1])["feedback"]["tiles"])
        text = summ.render(on_disk)
        self.assertIn("wrong_id", text)
        self.assertIn("id 5 -> 6", text)
        self.assertIn("MISSED (3.0, 4.0) true id 9", text)
        self.assertEqual(self.req("/api/feedback", {"tiles": {"nope": tile}})[0], 400)
        self.assertEqual(self.req("/api/feedback", {"tiles": {"strip_a_00": {"verdict": "bad"}}})[0], 400)

    def test_host_and_origin_checks(self):
        self.assertEqual(self.req("/api/state", headers={"Host": "evil.example"})[0], 403)
        self.assertEqual(self.req("/api/feedback", {"tiles": {}}, headers={"Origin": "http://evil.example"})[0], 403)

    def test_create_run_only_with_allowlisted_binary(self):
        self.assertEqual(self.req("/api/runs", {"name": "bad", "binary": "/bin/sh", "args": "-c id"})[0], 403)
        self.assertEqual(self.req("/api/runs", {"name": "bad", "binary": str(self.fake.resolve()), "args": "--out /tmp/x"})[0], 400)
        code, body, _ = self.req("/api/runs", {"name": "fakerun", "binary": str(self.fake.resolve()), "args": "--marker-diameter 32"})
        self.assertEqual(code, 202, body)
        run = self.app.runs.runs["fakerun"]
        for _ in range(100):
            if run.status["state"] != "running":
                break
            threading.Event().wait(0.1)
        self.assertEqual(run.status["state"], "done", run.status)
        manifest = json.loads((self.out / "runs" / "fakerun" / "manifest.json").read_text())
        self.assertEqual((manifest["args"], manifest["version"], manifest["complete"]), (["--marker-diameter", "32"], "fake-detector 9.9", True))
        d = json.loads(self.req("/api/run/fakerun/tile/strip_b_01")[1])
        self.assertEqual(d["n_decoded"], 2)
        code, body, _ = self.req("/api/compare/strip_a_00?a=fakerun&b=imp")
        self.assertEqual(code, 200)
        self.assertIn("counts", json.loads(body))
        self.assertEqual(self.req("/api/runs", {"name": "fakerun", "binary": str(self.fake.resolve()), "args": "--other"})[0], 409)


@unittest.skipIf(Image is None, "Pillow not available")
class AppBundle(unittest.TestCase):
    """The built front end is served from its directory only, with a CSP that
    allows nothing but 'self'; without a build the legacy page is served."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp()).resolve()
        data = cls.tmp / "data"
        data.mkdir()
        _png(data / "img.png")
        out = cls.tmp / "out"
        out.mkdir()
        cls.dist = cls.tmp / "dist"
        (cls.dist / "assets").mkdir(parents=True)
        (cls.dist / "index.html").write_text('<!doctype html><script type="module" src="/assets/app.js"></script>')
        (cls.dist / "assets" / "app.js").write_text("console.log(1)")
        (cls.dist / "assets" / "plex.woff2").write_bytes(b"wOF2")
        (cls.dist / "notes.json").write_text("{}")
        (cls.tmp / "secret.js").write_text("secret")
        cls.empty = cls.tmp / "empty"
        cls.empty.mkdir()
        app = srv.App(srv.Dataset(data, 1, out), out, {})
        cls.servers = {}
        for name, app_dir in (("built", cls.dist), ("unbuilt", cls.empty)):
            server = srv.make_server(app, "127.0.0.1", 0, app_dir)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            cls.servers[name] = server

    @classmethod
    def tearDownClass(cls):
        for server in cls.servers.values():
            server.shutdown()
            server.server_close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def get(self, which, path):
        url = f"http://127.0.0.1:{self.servers[which].server_address[1]}{path}"
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                return resp.status, resp.read(), resp.headers
        except urllib.error.HTTPError as e:
            return e.code, e.read(), e.headers

    def test_built_app_index_and_assets(self):
        code, body, hdr = self.get("built", "/")
        self.assertEqual(code, 200)
        self.assertIn(b"/assets/app.js", body)
        csp = hdr["Content-Security-Policy"]
        self.assertIn("default-src 'none'", csp)
        self.assertIn("script-src 'self';", csp)
        self.assertIn("font-src 'self'", csp)
        self.assertNotIn("http", csp)
        code, body, hdr = self.get("built", "/assets/app.js")
        self.assertEqual((code, body), (200, b"console.log(1)"))
        self.assertTrue(hdr["Content-Type"].startswith("text/javascript"))
        self.assertEqual(self.get("built", "/assets/plex.woff2")[2]["Content-Type"], "font/woff2")
        self.assertEqual(self.get("built", "/api/state")[0], 200)

    def test_built_app_never_serves_outside_or_unknown_files(self):
        for path in (
            "/assets/..%2f..%2fsecret.js", "/assets/../../secret.js", "/..%2fsecret.js", "/%2e%2e/secret.js",
            "/assets", "/notes.json", "/missing.js", "/assets/app.js%00.js", "/assets%5c..%5c..%5csecret.js",
        ):
            code, body, _ = self.get("built", path)
            self.assertEqual(code, 404, path)
            self.assertNotIn(b"secret", body)

    def test_unbuilt_app_falls_back_to_the_legacy_page(self):
        code, body, hdr = self.get("unbuilt", "/")
        self.assertEqual(code, 200)
        self.assertIn(b"ringgrid review", body)
        self.assertEqual(hdr["Content-Security-Policy"], srv.LEGACY_CSP)
        self.assertEqual(self.get("unbuilt", "/assets/app.js")[0], 404)
        self.assertFalse(srv.app_built(self.empty))
        self.assertFalse(srv.app_built(None))


@unittest.skipIf(Image is None, "Pillow not available")
class SharedSplit(unittest.TestCase):
    def test_review_tiles_match_eval_split(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        img = Image.new("L", (12, 4))
        img.putdata([x * 20 for _ in range(4) for x in range(12)])
        strip = tmp / "target_9.png"
        img.save(strip)
        import strip_tiles

        files = strip_tiles.split_strip(strip, tmp, 6)
        one = tmp / "one.png"
        strip_tiles.save_tile(strip, 3, 6, one)
        self.assertEqual(files[3].name, "target_9_03.png")
        self.assertEqual(Image.open(one).tobytes(), Image.open(files[3]).tobytes())


if __name__ == "__main__":
    unittest.main()
