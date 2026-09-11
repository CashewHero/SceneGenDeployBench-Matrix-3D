"""Matrix-3D panorama-to-3DGS generator. Model imports stay in job processes."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

from runner_wrapper.assets import PROFILES, ensure_assets
from runner_wrapper.job_logging import run_logged_command, tee_job_output
from runner_wrapper.measurements import ResourceMonitor
from runner_wrapper.pipeline import reconstruct


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name, str(int(default))).strip().lower()
    if value not in {"0", "1", "true", "false"}:
        raise ValueError(f"{name} must be 0, 1, true, or false")
    return value in {"1", "true"}


def configuration() -> dict:
    model = os.environ.get("MATRIX3D_VIDEO_MODEL", "5b-720p")
    if model not in PROFILES:
        raise ValueError(f"MATRIX3D_VIDEO_MODEL must be one of {', '.join(PROFILES)}")
    low_vram = env_bool("MATRIX3D_VRAM_MANAGEMENT", False)
    smoke_test = env_bool("MATRIX3D_SMOKE_TEST", False)
    if smoke_test and (model != "5b-720p" or not low_vram):
        raise ValueError("MATRIX3D_SMOKE_TEST requires the 5b-720p model with MATRIX3D_VRAM_MANAGEMENT=1")
    return {"video_model": model, "low_vram": low_vram, "smoke_test": smoke_test,
            "reconstructor": "opt", "attention_backend": "torch-sdpa"}


def parameters(raw: object) -> dict:
    defaults = {"prompt": "a realistic scene", "seed": 0, "angle": 0.0,
                "movement_range": 0.6, "movement_mode": "straight",
                "video_steps": 50, "gs_iterations": 3000}
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("job.parameters must be an object")
    unknown = set(raw) - defaults.keys()
    if unknown:
        raise ValueError(f"Unknown job parameters: {', '.join(sorted(unknown))}. "
                         "Select the model and low-VRAM mode through launcher.env.")
    result = defaults | raw
    if not isinstance(result["prompt"], str) or not result["prompt"].strip():
        raise ValueError("prompt must be a non-empty string")
    for name, minimum, maximum in (("seed", 0, 2**32 - 1), ("video_steps", 1, 1000),
                                   ("gs_iterations", 1, 30000)):
        value = result[name]
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    for name, minimum, maximum in (("angle", -360, 360), ("movement_range", 0, 0.8)):
        value = result[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be a finite number in [{minimum}, {maximum}]")
    if result["movement_range"] == 0:
        raise ValueError("movement_range must be positive; reconstruction normalizes camera displacement")
    if result["movement_mode"] not in ("straight", "s_curve", "l_curve", "r_curve"):
        raise ValueError("movement_mode must be straight, s_curve, l_curve, or r_curve")
    return result


def variant_key(config: dict, params: dict) -> str:
    encoded = json.dumps([config, params], sort_keys=True, allow_nan=False).encode()
    mode = "lowvram" if config["low_vram"] else "normal"
    if config["smoke_test"]:
        mode += "-smoke"
    return f"{config['video_model']}-{mode}-{hashlib.sha256(encoded).hexdigest()[:10]}"


def require_cuda(config: dict) -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Matrix-3D requires an NVIDIA CUDA GPU")
    major, minor = torch.cuda.get_device_capability(0)
    if config["smoke_test"] and (major, minor) < (7, 5):
        raise RuntimeError("Matrix-3D smoke mode requires CUDA compute capability 7.5 or newer")
    if not config["smoke_test"] and major < 8:
        raise RuntimeError("Matrix-3D uses bfloat16 video inference and requires an Ampere or newer GPU. "
                           "Use MATRIX3D_SMOKE_TEST=1 with the 5B low-VRAM profile for a plumbing-only Turing test.")


def prepare_image(request: dict, destination: Path) -> tuple[str, Path]:
    from PIL import Image
    job = request["job"]
    if job.get("job_type") not in ("generation", "generator"):
        raise ValueError("Matrix-3D accepts generation jobs only")
    primary = job["primary_sample"]
    samples = request.get("inputs", {}).get("data", {})
    if set(samples) != {primary}:
        raise ValueError("Matrix-3D requires exactly one primary sample in inputs.data")
    if request.get("inputs", {}).get("candidate") or request.get("inputs", {}).get("references"):
        raise ValueError("Matrix-3D does not consume candidate or reference inputs")
    path = Path(samples[primary]["image"])
    metadata = job.get("primary_sample_metadata") or {}
    if metadata.get("projection") != "equirectangular":
        raise ValueError("primary_sample_metadata.projection must be equirectangular; perspective input is not supported")
    if metadata.get("fov") is not None and metadata["fov"] != [360, 180]:
        raise ValueError("Matrix-3D requires a full 360 by 180 degree panorama")
    with Image.open(path) as image:
        if image.width != 2 * image.height:
            raise ValueError("The equirectangular image must have a 2:1 aspect ratio")
        image.convert("RGB").save(destination)
    return primary, path


def video_command(workspace: Path, config: dict, params: dict) -> list[str]:
    profile = PROFILES[config["video_model"]]
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1",
               str(workspace / "code/panoramic_image_to_video.py"),
               "--inout_dir", str(workspace / "pipeline"), "--resolution", str(profile["resolution"]),
               "--seed", str(params["seed"]), "--angle", str(params["angle"]),
               "--movement_range", str(params["movement_range"]), "--movement_mode", params["movement_mode"],
               "--num_inference_steps", str(params["video_steps"])]
    if config["video_model"] == "5b-720p":
        command.append("--use_5b_model")
    if config["low_vram"]:
        command.append("--enable_vram_management")
    if config["smoke_test"]:
        command.extend(["--num_frames", "5", "--width", "512", "--height", "256",
                        "--inference_dtype", "float16"])
        command[command.index("--num_inference_steps") + 1] = "1"
    return command


def utc(timestamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


def run_job(request: dict) -> dict:
    started = time.time()
    workspace = Path(request["runtime"]["workspace_dir"])
    workspace.mkdir(parents=True, exist_ok=True)
    fallback = hashlib.sha256(str(request.get("job", {}).get("job_id", "job")).encode()).hexdigest()[:10]
    try:
        config, params = configuration(), parameters(request["job"].get("parameters"))
        variant = variant_key(config, params)
    except (ValueError, KeyError):
        variant = f"invalid-{fallback}"
    log_path = workspace / f"runner-{variant}.log"
    report_path = workspace / f"metrics-{variant}.json"
    stage = "validation"
    monitor = None
    metrics = []
    report = {"inputs": request.get("inputs", {})}
    with tee_job_output(log_path):
        try:
            config, params = configuration(), parameters(request["job"].get("parameters"))
            report.update(parameters=params, model_configuration=config)
            pipeline = workspace / "pipeline"
            pipeline.mkdir()
            primary, image_path = prepare_image(request, pipeline / "pano_img.png")
            monitor = ResourceMonitor(sample_data={"image": str(image_path)}, output_dir=pipeline)
            monitor.start()
            require_cuda(config)
            repo = Path(__file__).resolve().parents[1]
            (workspace / "code").symlink_to(repo / "code", target_is_directory=True)
            (pipeline / "prompt.txt").write_text(params["prompt"], encoding="utf-8")
            stage = "model_assets"
            checkpoint_dir, identities = ensure_assets(workspace, config["video_model"],
                                                       env_bool("MATRIX3D_AUTO_DOWNLOAD_WEIGHTS", True))
            report["model_assets"] = identities
            (workspace / "checkpoints").symlink_to(checkpoint_dir, target_is_directory=True)
            env = os.environ.copy()
            env.update(MATRIX3D_CHECKPOINT_DIR=str(checkpoint_dir), MATRIX3D_SKIP_DOWNLOAD="1",
                       MATRIX3D_OPENCLIP_PATH=str(checkpoint_dir / "open_clip/open_clip_pytorch_model.bin"),
                       MATRIX3D_SVD_PATH=str(checkpoint_dir / "svd"),
                       PYTHONUNBUFFERED="1", OPENCV_IO_ENABLE_OPENEXR="1", PYOPENGL_PLATFORM="egl",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TQDM_MININTERVAL="10")
            # Pinned upstream StableSR checkpoints contain legacy Lightning objects.
            env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
            for name in ("XDG_CACHE_HOME", "TORCH_HOME", "HF_HOME", "MODELSCOPE_CACHE", "MPLCONFIGDIR"):
                env[name] = str(workspace / "runtime-cache" / name.lower())
            env["PYTHONPATH"] = os.pathsep.join(map(str, [repo, repo / "code", repo / "code/StableSR",
                                                         repo / "code/MoGe", repo / "code/DiffSynth-Studio"]))

            def run(name: str, command: list[str], cwd: Path = workspace) -> None:
                nonlocal stage
                stage = name
                print(f"Matrix-3D stage {name}", flush=True)
                run_logged_command(command, cwd=cwd, env=env)

            run("video_generation", video_command(workspace, config, params))
            source = reconstruct(workspace, checkpoint_dir, config, params, run)
            stage = "export"
            from runner_wrapper.geometry import export_scene
            output_name = f"3DGS-{variant}.ply"
            metadata = export_scene(source, workspace / output_name,
                                    pipeline / "geom_optim/data/normalization.json", params["angle"])
            outputs = {primary: {"3dgs": output_name}}
            report.update(output_files=outputs, output_metadata=metadata)
            result = {"status": "completed", "output_files": outputs, "output_metadata": metadata, "failure": None}
        except Exception as exc:
            traceback.print_exc()
            failure = {"code": "MATRIX3D_FAILED", "message": f"{stage}: {exc}",
                       "retryable": False, "stage": stage}
            report["failure"] = failure
            result = {"status": "failed", "failure": failure}
        finally:
            if monitor is not None:
                metrics = monitor.stop()
        report["resource_metrics"] = metrics
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    result.update(started_at=utc(started), completed_at=utc(time.time()), metrics=metrics,
                  artifacts=[{"artifact_type": "job_log", "path": log_path.name},
                             {"artifact_type": "metric_summary", "path": report_path.name}])
    return result
