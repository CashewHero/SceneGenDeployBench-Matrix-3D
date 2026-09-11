# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-venv build-essential ninja-build git patch ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/runner
ENV PATH=/opt/runner/bin:${PATH}
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9+PTX"
ARG MAX_JOBS=4
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} MAX_JOBS=${MAX_JOBS}
WORKDIR /build
COPY runner_wrapper/requirements.txt /build/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install pip==24.0 setuptools==75.8.0 wheel==0.45.1 ninja==1.11.1.4 && \
    pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126 && \
    pip install -r requirements.txt
COPY submodules/nvdiffrast /build/nvdiffrast
COPY submodules/simple-knn /build/simple-knn
COPY runner_wrapper/patches /build/patches
RUN test -f nvdiffrast/setup.py && patch -d nvdiffrast -p1 < patches/nvdiffrast-prebuilt.patch && \
    patch -d simple-knn -p1 < patches/simple-knn-cfloat.patch && \
    pip install --no-build-isolation --no-deps ./nvdiffrast ./simple-knn
COPY runner_wrapper/build_nvdiffrast.py /build/build_nvdiffrast.py
RUN python build_nvdiffrast.py build_ext --inplace && \
    cp nvdiffrast_plugin*.so /opt/runner/lib/python3.10/site-packages/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-build-isolation --no-deps \
      git+https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose.git@43e21bff91cd24986ee3dd52fe0bb06952e50ec7 \
      git+https://github.com/CompVis/taming-transformers.git@3ba01b241669f5ade541ce990f7650a3b8f65318
# Upstream find_packages() produces a metadata-only wheel because taming is a
# namespace package. Copy that exact revision's module tree into the runtime.
RUN git init /build/taming && \
    git -C /build/taming remote add origin https://github.com/CompVis/taming-transformers.git && \
    git -C /build/taming fetch --depth 1 origin 3ba01b241669f5ade541ce990f7650a3b8f65318 && \
    git -C /build/taming checkout --detach FETCH_HEAD && \
    cp -r /build/taming/taming /opt/runner/lib/python3.10/site-packages/taming
# Verify the extension architecture coverage, including forward-compatible PTX.
RUN cuobjdump --list-elf /opt/runner/lib/python3.10/site-packages/nvdiffrast_plugin*.so && \
    cuobjdump --list-ptx /opt/runner/lib/python3.10/site-packages/nvdiffrast_plugin*.so

FROM nvidia/cuda:12.6.3-base-ubuntu22.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 ca-certificates ffmpeg libgl1 libegl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/runner /opt/runner
WORKDIR /app/model_repo
ENV PATH=/opt/runner/bin:${PATH} \
    PYTHONPATH=/app/model_repo:/app/model_repo/code:/app/model_repo/code/StableSR:/app/model_repo/code/MoGe:/app/model_repo/code/DiffSynth-Studio \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENCV_IO_ENABLE_OPENEXR=1 PYOPENGL_PLATFORM=egl \
    RUNNER_PORT=58090 RUNNER_NAME=matrix3d-panorama-5b-opt RUNNER_TYPE=generator \
    RUNNER_VERSION=0.1.0 RUNNER_CONTRACT_VERSION=1 RUNNER_IDLE_TIMEOUT_SECONDS=900 \
    RUNNER_ADAPTER=runner_wrapper.adapter:run_job \
    MATRIX3D_VIDEO_MODEL=5b-720p MATRIX3D_VRAM_MANAGEMENT=0 MATRIX3D_AUTO_DOWNLOAD_WEIGHTS=1
COPY code /app/model_repo/code
COPY runner_wrapper /app/model_repo/runner_wrapper
RUN --network=none python -m runner_wrapper.check_imports
EXPOSE 58090
CMD ["python", "-m", "runner_wrapper.server"]
