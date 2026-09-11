"""No-weight, no-GPU import checks for every launched native entry point."""
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import nvdiffrast_plugin
import simple_knn._C
from diff_gaussian_rasterization import GaussianRasterizationSettings
from diffsynth import WanVideoPipeline
from diffsynth.pipelines.wan_video_new import WanVideoPipelineNew
from ldm.modules.encoders.modules import FrozenOpenCLIPEmbedder
from ldm.models.diffusion.ddpm import LatentDiffusionSRTextWT
from moge.model import MoGeModel
from open_clip.transformer import ResidualAttentionBlock
from runner_wrapper.adapter import run_job

assert "projmatrix_raw" in GaussianRasterizationSettings._fields

# StableSR was written for sequence-first OpenCLIP blocks. Current OpenCLIP
# blocks are batch-first, so exercise that compatibility path without weights.
embedder = FrozenOpenCLIPEmbedder.__new__(FrozenOpenCLIPEmbedder)
torch.nn.Module.__init__(embedder)
embedder.layer_idx = 0
embedder.model = SimpleNamespace(
    token_embedding=torch.nn.Embedding(8, 4),
    positional_embedding=torch.zeros(4, 4),
    transformer=SimpleNamespace(
        resblocks=torch.nn.ModuleList([ResidualAttentionBlock(4, 1)]),
        grad_checkpointing=False,
    ),
    attn_mask=torch.zeros(4, 4),
    ln_final=torch.nn.Identity(),
)
assert embedder.encode_with_transformer(torch.zeros(1, 4, dtype=torch.long)).shape == (1, 4, 4)

root = Path(__file__).resolve().parents[1]
for script in (
    "panoramic_image_to_video.py",
    "MoGe/scripts/infer_panorama.py",
    "VideoSR/scripts/enhance_video_pipeline.py",
    "utils_3dscene/panorama_video_to_perspective_depth_sequential.py",
    "utils_3dscene/gs_optim_datagen.py",
    "StableSR/scripts/sr_val_ddpm_text_T_vqganfin_old.py",
    "Pano_GS_Opt/train.py",
):
    print(f"Checking imports for {script}", flush=True)
    subprocess.run([sys.executable, str(root / "code" / script), "--help"],
                   cwd=root, check=True, stdout=subprocess.DEVNULL)
print("Matrix-3D imports passed", flush=True)
