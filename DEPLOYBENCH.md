# Matrix-3D generator

This wrapper accepts one full equirectangular `image` and produces an optimized `3dgs` PLY for SceneGenDeployBench. The input must be 2:1 with `job.primary_sample_metadata.projection: equirectangular`. Perspective-to-panorama generation and LRM reconstruction are not included.

## Model selection

Set the checkpoint and memory mode in catalog `launcher.env`, not job parameters. The [catalog](runner_wrapper/config/runners/matrix3d.yaml) provides separate runner identities for each checkpoint. All entries use the same image.

| Runner | `MATRIX3D_VIDEO_MODEL` | Default `MATRIX3D_VRAM_MANAGEMENT` |
| --- | --- | --- |
| `matrix3d-panorama-5b-opt` | `5b-720p` | `0` |
| `matrix3d-panorama-14b720-opt` | `14b-720p` | `1` |
| `matrix3d-panorama-14b480-opt` | `14b-480p` | `1` |

`MATRIX3D_VRAM_MANAGEMENT=1` enables upstream CPU offloading with zero persistent DiT parameters on GPU. `0` keeps the upstream normal memory policy. Either setting works with any listed model. This switch applies to video generation only, not MoGe, StableSR, VEnhancer or 3DGS optimization. The 480p profile adds VEnhancer, so it is not necessarily the cheapest complete pipeline. Low-VRAM mode trades GPU memory for RAM and transfer time; it does not guarantee a particular whole-pipeline VRAM limit.

Upstream reports about 19 GB for 5B 720p video generation, or 12 GB in low-VRAM mode. Its 14B figures are about 60/19 GB at 720p and 40/15 GB at 480p. Optimization reconstruction is listed at about 10 GB, while the repository states a 16 GB minimum for the complete pipeline. Treat these as upstream measurements, not limits enforced by this runner.

The adapter uses one visible CUDA GPU and native PyTorch SDPA attention. FlashAttention and distributed sequence parallelism are not installed. Production video inference requires bfloat16 on Ampere or newer hardware. Configure `launcher.gpus: device=<GPU UUID>` to choose a specific device. Do not expose the unstable GPU 0 on ws-3.

`MATRIX3D_SMOKE_TEST=1` is a local plumbing mode for the 11 GB RTX 2080 Ti. It requires `5b-720p` and low-VRAM mode, switches video inference to FP16, generates 5 frames at 512x256 with one diffusion step, runs depth alignment for every frame, and limits reconstruction to one optimization iteration with 20,000 initial points. Normal runner catalog entries never enable it. It executes the real checkpoint and every pipeline stage, but it does not test production dimensions, quality, performance, or peak VRAM. The full 5B checkpoint and common reconstruction weights are still required, about 47 GB in total. Initial atomic publication can temporarily need roughly twice that space.

Copy the catalog into the deployment's active runner-config directory. Its `0.1.0` image tag is a proposed release, not a published image; use the local image tag until publication. To compare normal and low-VRAM performance as separate benchmark runners, copy an entry under another `runner` name and change its environment. The model, mode and parameter hash also appear in output filenames and the metrics report.

## Job parameters

| Parameter | Default | Accepted values |
| --- | --- | --- |
| `prompt` | `a realistic scene` | Non-empty text describing the panorama |
| `seed` | `0` | Integer from 0 to 4294967295 |
| `angle` | `0` | Movement yaw in degrees, -360 to 360 |
| `movement_range` | `0.6` | Greater than 0 and at most 0.8, relative to estimated depth |
| `movement_mode` | `straight` | `straight`, `s_curve`, `l_curve`, `r_curve` |
| `video_steps` | `50` | Integer from 1 to 1000 |
| `gs_iterations` | `3000` | Integer from 1 to 30000 |

The native video length stays at 81 frames. The [smoke request](runner_wrapper/examples/generator_job_request.json) uses one video step and one optimization iteration to test plumbing, not quality. It still runs the other reconstruction stages and loads full models. StableSR retains its upstream fixed seed of 42; the job seed controls video generation.

## Weights and outputs

[assets.py](runner_wrapper/assets.py) is the source of truth for pinned repositories, revisions and filenames. Each profile downloads its selected Wan base model and Matrix-3D LoRA, plus MoGe, StableSR and OpenCLIP. The 480p profile also needs VEnhancer and the SVD VAE. No weights are bundled in the image.

`MATRIX3D_AUTO_DOWNLOAD_WEIGHTS=1` downloads missing assets. Set it to `0` to require a warm cache. `MATRIX3D_MODEL_CACHE_NAMESPACE` defaults to `matrix3d`, beneath `PATH_MODEL_CACHE`. Downloads stage in the job workspace, then publish under locked, revision-specific cache entries. Different jobs reuse common assets. Reserve space for both staged downloads and the published cache; these models require many gigabytes. Supply `HF_TOKEN` through deployment environment passthrough if access requires it. Model subprocesses run offline after preparation. Legacy PyTorch checkpoint loading is enabled only for these pinned upstream assets.

Only the fused 3DGS PLY is a reusable output. The wrapper undoes camera-centering, distance normalization and panorama yaw before exporting, including Gaussian scales and rotations. Output metadata declares `RDF` coordinates, primary-viewpoint origin and relative units. No metric `scene_scale` is invented; calibrate dataset-to-scene translation before evaluating novel viewpoints. Logs and metrics JSON are artifacts. Intermediate videos, depth and optimization data stay in the temporary job workspace.

The HTTP lifecycle and publication contract are unchanged. See [Runner API](runner_wrapper/docs/api.md).

## Local checks

Initialize the pinned submodule before building, then install only the small test dependencies on the host.

```bash
git submodule update --init --recursive
python3 -m venv .venv
.venv/bin/pip install -r runner_wrapper/requirements-test.txt
PYTHON_BIN=.venv/bin/python runner_wrapper/localtest.sh test
PYTHON_BIN=.venv/bin/python runner_wrapper/localtest.sh build
```

The multi-stage CUDA image builds extensions for SM 7.5, 8.0, 8.6 and 8.9 with forward-compatible PTX. The runtime image does not need a compiler. Build checks import all pipeline components without loading weights.

Use a real panorama on a supported GPU. Set a persistent data directory before allowing downloads.

```bash
export RUNNER_DATA_DIR=/mnt/nvme1/deploybench/matrix3d-localtest
export RUNNER_INPUT_IMAGE=/absolute/path/to/panorama.png
export MATRIX3D_VIDEO_MODEL=5b-720p
export MATRIX3D_VRAM_MANAGEMENT=1
export MATRIX3D_SMOKE_TEST=1
runner_wrapper/localtest.sh smoke
runner_wrapper/localtest.sh status
runner_wrapper/localtest.sh logs
runner_wrapper/localtest.sh down
```

`smoke` submits and returns without waiting for inference. It does not rebuild the image. `down` removes only the labeled test container and leaves data intact. `RUNNER_GPUS`, `RUNNER_IMAGE`, `RUNNER_HOST_PORT` and `RUNNER_REQUEST_FILE` override local defaults. Check for `finished`, a nonempty `3dgs` output, log and metrics artifacts, then render the output at the primary viewpoint to validate orientation. Full GPU generation and render validation remain required before publication.

## Integration changes

Local validation on 2026-09-10 passed the Docker build, offline imports for every pipeline entry point, 29 unit tests, nvdiffrast and simple-knn CUDA execution, and Gaussian rasterizer forward/backward execution. An HTTP smoke request with a real TartanAir panorama reached the expected unsupported-GPU failure on the RTX 2080 Ti, publishing logs and resource metrics without downloading weights. Successful end-to-end generation and render validation on a supported GPU are still pending.

Small native-code changes make checkpoint paths explicit, honor the 5B seed, expose video step count, avoid unnecessary single-GPU xfuser imports, use the installed ffmpeg, and record the inverse scene normalization. StableSR accepts an explicit VQGAN config; the wrapper derives an inference config without its unused training-only LPIPS loss or VGG download. The scene saver exports its existing fused-Gaussian representation for standard renderers. Reconstruction stages and nested MoGe calls use checked return codes so a failed native subprocess cannot silently produce a successful job.
