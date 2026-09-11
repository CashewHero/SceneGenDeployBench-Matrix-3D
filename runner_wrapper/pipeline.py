"""Checked launch commands for the upstream optimization reconstruction stages."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

from runner_wrapper.assets import PROFILES


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Model stage did not produce {path}")


def reconstruct(workspace: Path, checkpoints: Path, config: dict, params: dict, run) -> Path:
    pipeline = workspace / "pipeline"
    video = pipeline / "generated/generated.mp4"
    condition = pipeline / "condition"
    for path in (video, condition / "cameras.npz", condition / "firstframe_depth.exr", condition / "firstframe_mask.png"):
        require_file(path)
    code = workspace / "code"
    smoke_test = config["smoke_test"]
    width, height = (512, 256) if smoke_test else (1440, 720)
    if PROFILES[config["video_model"]]["resolution"] == 480:
        run("video_super_resolution", [sys.executable, str(code / "VideoSR/scripts/enhance_video_pipeline.py"),
            "--input_path", str(video), "--model_path", str(checkpoints / "VEnhancer/venhancer_v2.pt"),
            "--version", "v2", "--up_scale", "2", "--target_fps", "20", "--noise_aug", "100",
            "--solver_mode", "fast", "--steps", "15", "--prompt", params["prompt"],
            "--save_dir", str(video.parent), "--suffix", "enhancement"])
        video = video.with_name("generated_resize_enhance.mp4")
        require_file(video)
        width, height = 1920, 960
    geometry = pipeline / "geom_optim"
    data = geometry / "data"
    run("depth_alignment", [sys.executable, str(code / "utils_3dscene/panorama_video_to_perspective_depth_sequential.py"),
        "--device", "cuda:0", "--camera_path", str(condition / "cameras.npz"), "--video_path", str(video),
        "--anchor_frame_depth_paths", str(condition / "firstframe_depth.exr"),
        "--anchor_frame_mask_paths", str(condition / "firstframe_mask.png"), "--anchor_frame_indices", "0",
        "--output_dir", str(geometry), "--depth_estimation_interval", "1" if smoke_test else "10",
        "--width", str(width), "--height", str(height)])
    run("perspective_conversion", [sys.executable, str(code / "utils_3dscene/gs_optim_datagen.py"),
        "--optimized_depth_dir", str(data / "optimized_depths"), "--camera_path", str(condition / "cameras.npz"),
        "--output_dir", str(data)])
    require_file(data / "normalization.json")
    (data / "mv_rgb").rename(data / "mv_rgb_ori")
    # Training-only LPIPS would download VGG weights, but is never used for inference.
    vqgan = yaml.safe_load((code / "StableSR/configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml").read_text())
    vqgan["model"]["params"]["lossconfig"] = {"target": "torch.nn.Identity"}
    vqgan_path = pipeline / "vqgan-inference.yaml"
    vqgan_path.write_text(yaml.safe_dump(vqgan))
    run("image_super_resolution", [sys.executable, str(code / "StableSR/scripts/sr_val_ddpm_text_T_vqganfin_old.py"),
        "--init-img", str(data / "mv_rgb_ori"), "--outdir", str(data / "mv_rgb"),
        "--config", str(code / "StableSR/configs/stableSRNew/v2-finetune_text_T_512.yaml"),
        "--ckpt", str(checkpoints / "StableSR/stablesr_turbo.ckpt"), "--ddpm_steps", "4", "--dec_w", "0.5",
        "--seed", "42", "--n_samples", "1", "--vqgan_ckpt", str(checkpoints / "StableSR/vqgan_cfw_00011.ckpt"),
        "--vqgan_config", str(vqgan_path),
        "--colorfix_type", "wavelet"])
    original_names = {p.name for p in (data / "mv_rgb_ori").glob("*.png")}
    generated_names = {p.name for p in (data / "mv_rgb").glob("*.png")}
    if not original_names or original_names != generated_names:
        raise RuntimeError("StableSR did not produce every perspective image")
    iterations = 1 if smoke_test else params["gs_iterations"]
    run("3dgs_optimization", [sys.executable, str(code / "Pano_GS_Opt/train.py"),
        "-s", str(data), "-m", str(geometry / "output"), "-r", "1", "--use_decoupled_appearance",
        "--save_iterations", str(iterations), "--test_iterations", str(iterations), "--sh_degree", "0",
        "--densify_from_iter", "500", "--densify_until_iter", "1501", "--iterations", str(iterations), "--eval",
        "--img_sample_interval", "1", "--num_views_per_view", "1" if smoke_test else "3",
        "--num_of_point_cloud", "20000" if smoke_test else "3000000",
        "--device", "cuda:0", "--distortion_from_iter", "6500", "--depth_normal_from_iter", "6500"])
    output = geometry / f"output/point_cloud/iteration_{iterations}/point_cloud_fused.ply"
    require_file(output)
    return output
