"""Build the pinned nvdiffrast CUDA plugin ahead of time, without a GPU."""
from pathlib import Path
import shutil

import nvdiffrast
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(nvdiffrast.__file__).parent
common = root / "common"
torch_dir = root / "torch"
sources = [common / name for name in (
    "cudaraster/impl/Buffer.cpp", "cudaraster/impl/CudaRaster.cpp", "cudaraster/impl/RasterImpl.cu",
    "cudaraster/impl/RasterImpl.cpp", "common.cpp", "rasterize.cu", "interpolate.cu", "texture.cu",
    "texture.cpp", "antialias.cu")]
sources += [torch_dir / name for name in (
    "torch_bindings.cpp", "torch_rasterize.cpp", "torch_interpolate.cpp", "torch_texture.cpp", "torch_antialias.cpp")]
# setuptools maps foo.cpp and foo.cu to the same object name. Preserve include
# lookup while giving the CUDA translation units distinct names in the builder.
for index, source in enumerate(sources):
    if source.suffix == ".cu":
        renamed = source.with_name(source.stem + "_cuda.cu")
        shutil.copyfile(source, renamed)
        sources[index] = renamed
setup(name="matrix3d-nvdiffrast-plugin", version="0.1.0",
      ext_modules=[CUDAExtension("nvdiffrast_plugin", list(map(str, sources)),
                                extra_compile_args={"cxx": ["-DNVDR_TORCH"], "nvcc": ["-DNVDR_TORCH"]})],
      cmdclass={"build_ext": BuildExtension})
