"""Preset cwm_render_parity: the walker training obs is the native MuJoCo render
physics.render(64, 64, camera_id=0), as TrajD/DRAMA's envs/my_dmc.DMControl, not CARL's
240x320 render squashed to 64x64 by PIL bilinear (ResizeImage).

CPU only, no GL: Physics.render is replaced by a stub that returns content keyed on
(height, width, camera_id). The test fails on 09ed30f (no preset -> 240x320 call + resize).

    CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD:$PWD/dreamerv3_compat:<CWM root> \\
        /home/jovyan/envs/dali/bin/python -m unittest contextual_mbrl.tests.test_native_render -v
"""
import os
import sys
import unittest
import warnings
from unittest import mock

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["MUJOCO_GL"] = "disable"
os.environ.pop("PYOPENGL_PLATFORM", None)
warnings.filterwarnings("ignore")
# DALI_TEST_REPO: run this test against another DALI tree (red check on 09ed30f).
REPO = os.environ.get("DALI_TEST_REPO") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (REPO, os.path.join(REPO, "dreamerv3_compat")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import ruamel.yaml as _yaml  # noqa: E402
from dm_control.mujoco import engine as _engine  # noqa: E402
from dreamerv3 import embodied  # noqa: E402

from contextual_mbrl.dreamer.envs import make_envs  # noqa: E402

# walker benches of record: DALI bench preset per bench (CWM benchmark/dali_bench_table.sh).
WALKER = {"walker_2f": "benchmark_regime_b", "walker_3f": "benchmark_regime_b_K3",
          "walker_mixed": "benchmark_regime_b_mixed"}
ARCH = "enc_img_dec_img_ctxencoder_gru_grssm"


def bench_config(bench):
    master = _yaml.YAML(typ="safe").load(
        open(os.path.join(REPO, "contextual_mbrl", "dreamer", "configs.yaml")))
    cfg = embodied.Config(master["defaults"])
    # Launcher order: carl, bench preset, arch preset, then the render preset (absent on 09ed30f).
    for name in ("carl", WALKER[bench], ARCH, "cwm_render_parity"):
        if name in master:
            cfg = cfg.update(master[name])
    return cfg.update({"seed": 11, "envs.amount": 1, "envs.parallel": "none",
                       "envs.restart": False, "jax.platform": "cpu"})


def _stub_pixels(height, width, camera_id):
    rng = np.random.RandomState(height * 100003 + width * 101 + camera_id + 7)
    return rng.randint(0, 256, (height, width, 3)).astype(np.uint8)


class NativeRender(unittest.TestCase):

    def test_walker_obs_is_native_64x64_camera0_render(self):
        calls = []

        def render(self, height=240, width=320, camera_id=-1, *a, **kw):
            calls.append((height, width, camera_id))
            return _stub_pixels(height, width, camera_id)

        with mock.patch.object(_engine.Physics, "render", render):
            for bench in WALKER:
                with self.subTest(bench=bench):
                    batch = make_envs(bench_config(bench))
                    try:
                        env = batch._envs[0]
                        self.assertEqual(tuple(env.obs_space["image"].shape), (64, 64, 3))
                        act = {k: np.zeros(v.shape, v.dtype) for k, v in env.act_space.items()}
                        act["reset"] = True
                        for _ in range(3):  # reset frame + 2 decision steps
                            calls.clear()
                            obs = env.step(act)
                            act["reset"] = False
                            self.assertEqual(calls[-1], (64, 64, 0),
                                             "obs frame is not a native 64x64 camera-0 render")
                            np.testing.assert_array_equal(obs["image"], _stub_pixels(64, 64, 0))
                    finally:
                        batch.close()


if __name__ == "__main__":
    unittest.main()
