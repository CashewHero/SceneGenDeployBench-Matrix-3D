"""Pinned, profile-specific downloads published atomically into the shared cache."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

from runner_wrapper.files import publish_directory

PROFILES = {
    "5b-720p": {"resolution": 720, "repo": "Wan-AI/Wan2.2-TI2V-5B", "shards": 3,
                "revision": "921dbaf3f1674a56f47e83fb80a34bac8a8f203e", "vae": "Wan2.2_VAE.pth",
                "lora": "pano_video_gen_720p_5b.safetensors"},
    "14b-720p": {"resolution": 720, "repo": "Wan-AI/Wan2.1-I2V-14B-720P", "shards": 7,
                 "revision": "8823af45fcc58a8aa999a54b04be9abc7d2aac98", "vae": "Wan2.1_VAE.pth",
                 "lora": "pano_video_gen_720p.bin"},
    "14b-480p": {"resolution": 480, "repo": "Wan-AI/Wan2.1-I2V-14B-480P", "shards": 7,
                 "revision": "6b73f84e66371cdfe870c72acd6826e1d61cf279", "vae": "Wan2.1_VAE.pth",
                 "lora": "pano_video_gen_480p.ckpt"},
}


def asset_specs(model: str) -> list[dict]:
    profile = PROFILES[model]
    count = profile["shards"]
    files = [f"diffusion_pytorch_model-{i:05d}-of-{count:05d}.safetensors" for i in range(1, count + 1)]
    files += [profile["vae"], "models_t5_umt5-xxl-enc-bf16.pth", "google/umt5-xxl/spiece.model",
              "google/umt5-xxl/tokenizer_config.json", "google/umt5-xxl/special_tokens_map.json"]
    if count == 7:
        files += ["models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"]
    specs = [
        {"repo": profile["repo"], "revision": profile["revision"], "directory": profile["repo"],
         "files": {name: name for name in files}},
        {"repo": "Skywork/Matrix-3D", "revision": "9f23eda5300f2a44d1256ecaf54a7808e69d7daa",
         "directory": f"loras/{model}", "files": {f"checkpoints/{profile['lora']}": profile["lora"]}},
        {"repo": "Ruicheng/moge-vitl", "revision": "ad326bfb61facd6c52b5a825bc1e34d7c97d9672",
         "directory": "moge", "files": {"model.pt": "model.pt"}},
        {"repo": "Iceclear/StableSR", "revision": "932e2863fb9f40f111d1c03c160a5fc94921cb62",
         "directory": "StableSR", "files": {name: name for name in ("stablesr_turbo.ckpt", "vqgan_cfw_00011.ckpt")}},
        {"repo": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "revision": "1c2b8495b28150b8a4922ee1c8edee224c284c0c",
         "directory": "open_clip", "files": {"open_clip_pytorch_model.bin": "open_clip_pytorch_model.bin"}},
    ]
    if profile["resolution"] == 480:
        specs.extend([
            {"repo": "jwhejwhe/VEnhancer", "revision": "ac52f5b8526a4504a085c20061e03ce0771ffdf7",
             "directory": "VEnhancer", "files": {"venhancer_v2.pt": "venhancer_v2.pt"}},
            {"repo": "stabilityai/stable-video-diffusion-img2vid", "revision": "9cf024d5bfa8f56622af86c884f26a52f6676f2e",
             "directory": "svd", "files": {name: name for name in ("vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors")}},
        ])
    return specs


def _complete(path: Path, spec: dict) -> bool:
    try:
        if json.loads((path / "asset.json").read_text()) != spec:
            return False
        return all((path / name).is_file() and (path / name).stat().st_size > 0 for name in spec["files"].values())
    except (OSError, ValueError):
        return False


@contextmanager
def _download_cache(workspace: Path):
    cache = workspace / "model-downloads/.cache"
    values = {
        "XDG_CACHE_HOME": str(cache),
        "HF_HOME": str(cache / "huggingface"),
        "HF_HUB_CACHE": str(cache / "huggingface/hub"),
        "HF_XET_CACHE": str(cache / "huggingface/xet"),
    }
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def ensure_assets(workspace: Path, model: str, auto_download: bool) -> tuple[Path, list[dict]]:
    namespace = os.getenv("MATRIX3D_MODEL_CACHE_NAMESPACE", "matrix3d")
    if not namespace or namespace in {".", ".."} or Path(namespace).name != namespace:
        raise ValueError("MATRIX3D_MODEL_CACHE_NAMESPACE must be one directory name")
    root = Path(os.getenv("PATH_MODEL_CACHE", "/data/model_cache")) / namespace
    root.mkdir(parents=True, exist_ok=True)
    layout = workspace / "model-layout"
    layout.mkdir()
    specs = asset_specs(model)
    with _download_cache(workspace):
        for spec in specs:
            key = spec["repo"].replace("/", "--") + "-" + spec["revision"]
            key += "-" + hashlib.sha256(json.dumps(spec["files"], sort_keys=True).encode()).hexdigest()[:10]
            destination = root / key
            with (root / (key + ".lock")).open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if not _complete(destination, spec):
                    if not auto_download:
                        raise FileNotFoundError(f"Missing pinned model assets in {destination}; enable MATRIX3D_AUTO_DOWNLOAD_WEIGHTS")
                    from huggingface_hub import hf_hub_download
                    staging = workspace / "model-downloads" / key
                    staging.mkdir(parents=True, exist_ok=True)
                    for filename, target in spec["files"].items():
                        downloaded = Path(hf_hub_download(repo_id=spec["repo"], revision=spec["revision"],
                                                           filename=filename, local_dir=staging / "download"))
                        output = staging / target
                        output.parent.mkdir(parents=True, exist_ok=True)
                        downloaded.rename(output)
                    shutil.rmtree(staging / "download")
                    (staging / "asset.json").write_text(json.dumps(spec, indent=2) + "\n")
                    publish_directory(staging, destination, dirs_exist_ok=True)
                    if not _complete(destination, spec):
                        raise RuntimeError(f"Model asset publication failed: {destination}")
                    shutil.rmtree(staging)
            link = layout / spec["directory"]
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(destination, target_is_directory=True)
    (layout / "Wan-AI/wan_lora").symlink_to(layout / f"loras/{model}", target_is_directory=True)
    return layout, specs
