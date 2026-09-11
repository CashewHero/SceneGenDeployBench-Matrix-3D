from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml
from PIL import Image
from plyfile import PlyData, PlyElement

from runner_wrapper import adapter, assets, geometry, pipeline

REPO = Path(__file__).resolve().parents[2]


def gaussian(path: Path) -> None:
    names = ["x", "y", "z", "opacity", "f_dc_0", "f_dc_1", "f_dc_2",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    vertices = np.zeros(1, dtype=[(name, "f4") for name in names])
    vertices["z"] = 1
    vertices["rot_0"] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")]).write(path)


def request(root: Path) -> dict:
    image = root / "input.png"
    Image.new("RGB", (64, 32)).save(image)
    return {"job": {"job_id": "test", "job_type": "generation", "primary_sample": "sample",
                    "primary_sample_metadata": {"projection": "equirectangular"}, "parameters": {}},
            "inputs": {"data": {"sample": {"image": str(image)}}},
            "runtime": {"workspace_dir": str(root / "workspace")}}


class ConfigurationTests(unittest.TestCase):
    def test_all_model_and_memory_combinations(self):
        params = adapter.parameters({"seed": 123, "video_steps": 7})
        variants = set()
        for model, profile in assets.PROFILES.items():
            for low in ("0", "1"):
                with self.subTest(model=model, low=low), patch.dict(os.environ, {
                    "MATRIX3D_VIDEO_MODEL": model, "MATRIX3D_VRAM_MANAGEMENT": low,
                }):
                    config = adapter.configuration()
                    cmd = adapter.video_command(Path("/work"), config, params)
                    self.assertEqual("--enable_vram_management" in cmd, low == "1")
                    self.assertEqual("--use_5b_model" in cmd, model == "5b-720p")
                    self.assertEqual(cmd[cmd.index("--resolution") + 1], str(profile["resolution"]))
                    self.assertEqual(cmd[cmd.index("--seed") + 1], "123")
                    self.assertEqual(cmd[cmd.index("--num_inference_steps") + 1], "7")
                    self.assertIn("--nproc_per_node=1", cmd)
                    variants.add(adapter.variant_key(config, params))
        self.assertEqual(len(variants), 6)

    def test_reduced_smoke_configuration(self):
        env = {"MATRIX3D_VIDEO_MODEL": "5b-720p", "MATRIX3D_VRAM_MANAGEMENT": "1",
               "MATRIX3D_SMOKE_TEST": "1"}
        with patch.dict(os.environ, env, clear=True):
            config = adapter.configuration()
            command = adapter.video_command(Path("/work"), config, adapter.parameters({"video_steps": 50}))
        self.assertTrue(config["smoke_test"])
        self.assertIn("-smoke-", adapter.variant_key(config, adapter.parameters({})))
        for name, expected in (("--num_frames", "5"), ("--width", "512"), ("--height", "256"),
                               ("--inference_dtype", "float16"), ("--num_inference_steps", "1")):
            self.assertEqual(command[command.index(name) + 1], expected)

        for bad in ({"MATRIX3D_VIDEO_MODEL": "14b-720p", "MATRIX3D_VRAM_MANAGEMENT": "1"},
                    {"MATRIX3D_VIDEO_MODEL": "5b-720p", "MATRIX3D_VRAM_MANAGEMENT": "0"}):
            with self.subTest(bad=bad), patch.dict(os.environ, bad | {"MATRIX3D_SMOKE_TEST": "1"}, clear=True), \
                    self.assertRaisesRegex(ValueError, "requires"):
                adapter.configuration()

    def test_invalid_configuration_and_parameters(self):
        for env in ({"MATRIX3D_VIDEO_MODEL": "bad"}, {"MATRIX3D_VRAM_MANAGEMENT": "sometimes"}):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True), self.assertRaises(ValueError):
                adapter.configuration()
        invalid = [{"seed": True}, {"seed": -1}, {"seed": 2**32}, {"angle": float("nan")},
                   {"angle": float("inf")}, {"movement_range": 0}, {"movement_range": .81},
                   {"movement_mode": "orbit"}, {"prompt": " "}, {"video_steps": 0},
                   {"gs_iterations": 30001}, {"video_model": "5b-720p"}, {"low_vram": True}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                adapter.parameters(value)

    def test_catalog_matches_adapter(self):
        catalog = yaml.safe_load((REPO / "runner_wrapper/config/runners/matrix3d.yaml").read_text())
        self.assertEqual(catalog["catalog_version"], 1)
        models = set()
        for entry in catalog["runners"]:
            self.assertEqual(entry["kind"], "generator")
            self.assertEqual(entry["inputs"]["data"]["required_sample"]["required_datatype"], ["image"])
            self.assertEqual(entry["job_parameters"], adapter.parameters({}))
            with patch.dict(os.environ, entry["launcher"]["env"], clear=True):
                models.add(adapter.configuration()["video_model"])
        self.assertEqual(models, set(assets.PROFILES))

    def test_input_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = request(root)
            primary, source = adapter.prepare_image(job, root / "prepared.png")
            self.assertEqual(primary, "sample")
            self.assertEqual(source, root / "input.png")
            job["job"]["primary_sample_metadata"]["projection"] = "perspective"
            with self.assertRaisesRegex(ValueError, "perspective"):
                adapter.prepare_image(job, root / "prepared.png")
            job["job"]["primary_sample_metadata"]["projection"] = "equirectangular"
            Image.new("RGB", (32, 32)).save(source)
            with self.assertRaisesRegex(ValueError, "2:1"):
                adapter.prepare_image(job, root / "prepared.png")


class AssetTests(unittest.TestCase):
    def test_only_selected_profile_is_downloaded(self):
        for model, profile in assets.PROFILES.items():
            specs = assets.asset_specs(model)
            repos = {spec["repo"] for spec in specs}
            self.assertEqual({repo for repo in repos if repo.startswith("Wan-AI/")}, {profile["repo"]})
            self.assertEqual("jwhejwhe/VEnhancer" in repos, model == "14b-480p")
            for spec in specs:
                self.assertRegex(spec["revision"], r"^[0-9a-f]{40}$")

    def test_published_cache_is_reused_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = [{"repo": "Wan-AI/test", "revision": "a" * 40, "directory": "Wan-AI/test",
                      "files": {"nested/weight.bin": "weight.bin"}},
                     {"repo": "test/lora", "revision": "b" * 40, "directory": "loras/5b-720p",
                      "files": {"lora.bin": "lora.bin"}}]

            def download(*, repo_id, revision, filename, local_dir):
                expected = root / str(index) / "model-downloads/.cache/huggingface/xet"
                self.assertEqual(os.environ["HF_XET_CACHE"], str(expected))
                path = Path(local_dir) / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"mock weight")
                return str(path)

            with patch.dict(os.environ, {"PATH_MODEL_CACHE": str(root / "cache")}, clear=True), \
                    patch.object(assets, "asset_specs", return_value=specs), \
                    patch("huggingface_hub.hf_hub_download", side_effect=download) as fetch:
                for index, auto in enumerate((True, False)):
                    work = root / str(index)
                    work.mkdir()
                    layout, actual = assets.ensure_assets(work, "5b-720p", auto)
                    self.assertEqual(actual, specs)
                    self.assertEqual((layout / "Wan-AI/test/weight.bin").read_bytes(), b"mock weight")
                    self.assertTrue((layout / "Wan-AI/wan_lora/lora.bin").is_file())
                self.assertEqual(fetch.call_count, 2)
                self.assertNotIn("HF_XET_CACHE", os.environ)

    def test_missing_assets_fail_offline(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"PATH_MODEL_CACHE": tmp}, clear=True):
            work = Path(tmp) / "work"
            work.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "Missing pinned model assets"):
                assets.ensure_assets(work, "14b-720p", False)

    def test_interrupted_download_does_not_publish_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            work.mkdir()
            with patch.dict(os.environ, {"PATH_MODEL_CACHE": str(root / "cache")}, clear=True), \
                    patch("huggingface_hub.hf_hub_download", side_effect=OSError("download interrupted")):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    assets.ensure_assets(work, "5b-720p", True)
            self.assertFalse(list((root / "cache").rglob("asset.json")))


class GeometryTests(unittest.TestCase):
    def test_inverse_normalization_and_yaw(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target, transform = root / "source.ply", root / "out.ply", root / "normalization.json"
            gaussian(source)
            transform.write_text(json.dumps({"center": [1, 2, 3], "scale": 2}))
            metadata = geometry.export_scene(source, target, transform, 90)
            vertex = PlyData.read(target)["vertex"].data[0]
            # Unnormalization gives (1, 2, 5), positive yaw maps forward to right.
            np.testing.assert_allclose([vertex[n] for n in ("x", "y", "z")], [5, 2, -1], atol=1e-6)
            np.testing.assert_allclose(vertex["scale_0"], np.log(2), atol=1e-6)
            np.testing.assert_allclose([vertex["rot_0"], vertex["rot_2"]], [2**-.5, 2**-.5], atol=1e-6)
            self.assertEqual(metadata["scene_coordinate_system"], "RDF")
            self.assertNotIn("scene_scale", metadata)

    def test_invalid_output_is_not_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target, transform = root / "source.ply", root / "out.ply", root / "normalization.json"
            gaussian(source)
            transform.write_text(json.dumps({"center": [0, 0, 0], "scale": 0}))
            with self.assertRaisesRegex(ValueError, "normalization"):
                geometry.export_scene(source, target, transform, 0)
            transform.write_text(json.dumps({"center": [0, 0, 0], "scale": 1}))
            PlyData([PlyElement.describe(np.zeros(1, dtype=[("x", "f4")]), "vertex")]).write(source)
            with self.assertRaisesRegex(ValueError, "Gaussian PLY"):
                geometry.export_scene(source, target, transform, 0)
            self.assertFalse(target.exists())


class PipelineTests(unittest.TestCase):
    def test_reconstruction_stages_and_export(self):
        for model in assets.PROFILES:
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data = root / "pipeline/geom_optim/data"
                (root / "code").symlink_to(REPO / "code", target_is_directory=True)
                for name in ("generated/generated.mp4", "condition/cameras.npz", "condition/firstframe_depth.exr", "condition/firstframe_mask.png"):
                    path = root / "pipeline" / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"stage fixture")
                stages = []

                def run(stage, command):
                    stages.append(stage)
                    self.assertEqual(command[0], pipeline.sys.executable)
                    if stage == "video_super_resolution":
                        (root / "pipeline/generated/generated_resize_enhance.mp4").write_bytes(b"video")
                    elif stage == "perspective_conversion":
                        (data / "mv_rgb").mkdir(parents=True)
                        (data / "mv_rgb/0000.png").write_bytes(b"image")
                        (data / "normalization.json").write_text('{"center": [0, 0, 0], "scale": 1}')
                    elif stage == "image_super_resolution":
                        vqgan_config = yaml.safe_load(Path(command[command.index("--vqgan_config") + 1]).read_text())
                        self.assertEqual(vqgan_config["model"]["params"]["lossconfig"], {"target": "torch.nn.Identity"})
                        (data / "mv_rgb").mkdir()
                        (data / "mv_rgb/0000.png").write_bytes(b"image")
                    elif stage == "3dgs_optimization":
                        gaussian(root / "pipeline/geom_optim/output/point_cloud/iteration_1/point_cloud_fused.ply")

                output = pipeline.reconstruct(root, root / "weights", {"video_model": model, "smoke_test": False},
                                              adapter.parameters({"gs_iterations": 1}), run)
                self.assertTrue(output.is_file())
                expected = ["depth_alignment", "perspective_conversion", "image_super_resolution", "3dgs_optimization"]
                if model == "14b-480p":
                    expected.insert(0, "video_super_resolution")
                self.assertEqual(stages, expected)

    def test_missing_video_stops_reconstruction(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(pipeline, "require_file", wraps=pipeline.require_file):
            with self.assertRaises(FileNotFoundError):
                pipeline.reconstruct(Path(tmp), Path(tmp), {"video_model": "5b-720p", "smoke_test": False}, {}, self.fail)

    def test_smoke_reconstruction_reduces_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "pipeline/geom_optim/data"
            (root / "code").symlink_to(REPO / "code", target_is_directory=True)
            for name in ("generated/generated.mp4", "condition/cameras.npz", "condition/firstframe_depth.exr",
                         "condition/firstframe_mask.png"):
                path = root / "pipeline" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            commands = {}

            def run(stage, command):
                commands[stage] = command
                if stage == "perspective_conversion":
                    (data / "mv_rgb").mkdir(parents=True)
                    (data / "mv_rgb/0000.png").write_bytes(b"image")
                    (data / "normalization.json").write_text('{"center": [0, 0, 0], "scale": 1}')
                elif stage == "image_super_resolution":
                    (data / "mv_rgb").mkdir()
                    (data / "mv_rgb/0000.png").write_bytes(b"image")
                elif stage == "3dgs_optimization":
                    gaussian(root / "pipeline/geom_optim/output/point_cloud/iteration_1/point_cloud_fused.ply")

            pipeline.reconstruct(root, root / "weights", {"video_model": "5b-720p", "smoke_test": True},
                                 adapter.parameters({}), run)
            depth = commands["depth_alignment"]
            self.assertEqual(depth[depth.index("--width") + 1], "512")
            self.assertEqual(depth[depth.index("--height") + 1], "256")
            self.assertEqual(depth[depth.index("--depth_estimation_interval") + 1], "1")
            optimize = commands["3dgs_optimization"]
            self.assertEqual(optimize[optimize.index("--iterations") + 1], "1")
            self.assertEqual(optimize[optimize.index("--num_of_point_cloud") + 1], "20000")


class AdapterTests(unittest.TestCase):
    def test_success_and_subprocess_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                job = request(root)
                work = Path(job["runtime"]["workspace_dir"])

                def reconstruct(workspace, checkpoints, config, params, run):
                    output = workspace / "pipeline/scene.ply"
                    gaussian(output)
                    transform = workspace / "pipeline/geom_optim/data/normalization.json"
                    transform.parent.mkdir(parents=True)
                    transform.write_text('{"center": [0, 0, 0], "scale": 1}')
                    return output

                with patch.dict(os.environ, {}, clear=True), patch.object(adapter, "require_cuda"), \
                        patch.object(adapter, "ensure_assets", return_value=(root / "weights", [])) as ensure, \
                        patch.object(adapter, "ResourceMonitor") as monitor, \
                        patch.object(adapter, "reconstruct", side_effect=reconstruct), \
                        patch.object(adapter, "run_logged_command", side_effect=subprocess.CalledProcessError(1, "model") if fail else None) as run:
                    monitor.return_value.stop.return_value = []
                    result = adapter.run_job(job)
                    self.assertEqual(result["status"], "failed" if fail else "completed")
                    self.assertTrue(ensure.call_args.args[2])
                    self.assertEqual(run.call_args.kwargs["env"]["HF_HUB_OFFLINE"], "1")
                    self.assertEqual(run.call_args.kwargs["cwd"], work)
                    for artifact in result["artifacts"]:
                        self.assertTrue((work / artifact["path"]).is_file())
                    if fail:
                        self.assertNotIn("output_files", result)
                        self.assertEqual(result["failure"]["stage"], "video_generation")
                    else:
                        output = work / result["output_files"]["sample"]["3dgs"]
                        self.assertTrue(output.is_file())
                        report = json.loads((work / result["artifacts"][1]["path"]).read_text())
                        self.assertEqual(report["output_files"], result["output_files"])
                        self.assertEqual(report["model_configuration"]["video_model"], "5b-720p")
                    monitor.return_value.stop.assert_called_once()

    def test_auto_download_can_be_disabled(self):
        with patch.dict(os.environ, {"MATRIX3D_AUTO_DOWNLOAD_WEIGHTS": "0"}, clear=True):
            self.assertFalse(adapter.env_bool("MATRIX3D_AUTO_DOWNLOAD_WEIGHTS", True))

    def test_configuration_failure_has_logs_and_no_outputs(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"MATRIX3D_VIDEO_MODEL": "invalid"}):
            job = request(Path(tmp))
            result = adapter.run_job(job)
            self.assertEqual(result["status"], "failed")
            self.assertNotIn("output_files", result)
            self.assertEqual(result["failure"]["stage"], "validation")
            self.assertEqual(len(result["artifacts"]), 2)


if __name__ == "__main__":
    unittest.main()
