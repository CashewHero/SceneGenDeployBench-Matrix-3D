#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
IMAGE="${RUNNER_IMAGE:-scenegendeploybench-matrix-3d:local}"
CONTAINER="${RUNNER_CONTAINER:-matrix3d-localtest}"
HOST_PORT="${RUNNER_HOST_PORT:-58090}"
DATA_DIR="${RUNNER_DATA_DIR:-${TMPDIR:-/tmp}/matrix3d-localtest-data}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REQUEST_FILE="${RUNNER_REQUEST_FILE:-${SCRIPT_DIR}/examples/generator_job_request.json}"

run_tests() {
  PYTHONPATH="${REPO_ROOT}" "${PYTHON_BIN}" -m unittest discover -s "${SCRIPT_DIR}/tests" -v
}

run_container() {
  if docker container inspect "${CONTAINER}" >/dev/null 2>&1; then
    echo "Container ${CONTAINER} already exists. Inspect it or use down before starting another." >&2
    exit 1
  fi
  mkdir -p "${DATA_DIR}/datasets/smoke" "${DATA_DIR}/model_cache" "${DATA_DIR}/output" "${DATA_DIR}/pipelines"
  local env_args=(
    -e RUNNER_PORT=58090 -e RUNNER_TYPE=generator -e RUNNER_VERSION=0.1.0
    -e "RUNNER_NAME=matrix3d-localtest"
    -e PATH_DATASETS=/data/datasets -e PATH_MODEL_CACHE=/data/model_cache
    -e PATH_OUTPUT=/data/output -e PATH_PIPELINES=/data/pipelines
    -e "MATRIX3D_VIDEO_MODEL=${MATRIX3D_VIDEO_MODEL:-5b-720p}"
    -e "MATRIX3D_VRAM_MANAGEMENT=${MATRIX3D_VRAM_MANAGEMENT:-0}"
    -e "MATRIX3D_SMOKE_TEST=${MATRIX3D_SMOKE_TEST:-0}"
    -e "MATRIX3D_AUTO_DOWNLOAD_WEIGHTS=${MATRIX3D_AUTO_DOWNLOAD_WEIGHTS:-1}"
    -e "MATRIX3D_MODEL_CACHE_NAMESPACE=${MATRIX3D_MODEL_CACHE_NAMESPACE:-matrix3d}"
    -e HF_TOKEN
  )
  docker run -d --name "${CONTAINER}" --label deploybench.localtest=matrix3d \
    --gpus "${RUNNER_GPUS:-1}" --user "$(id -u):$(id -g)" \
    -p "127.0.0.1:${HOST_PORT}:58090" "${env_args[@]}" -v "${DATA_DIR}:/data" "${IMAGE}"
  for ((attempt=0; attempt<30; attempt++)); do
    if curl -fsS "http://127.0.0.1:${HOST_PORT}/status" 2>/dev/null; then
      echo
      return
    fi
    sleep 1
  done
  docker logs "${CONTAINER}" >&2
  echo "Runner did not become ready in 30 seconds." >&2
  exit 1
}

case "${1:-help}" in
  test) run_tests ;;
  build)
    run_tests
    docker build -f "${SCRIPT_DIR}/Dockerfile" -t "${IMAGE}" "${REPO_ROOT}"
    ;;
  run) run_container ;;
  smoke)
    : "${RUNNER_INPUT_IMAGE:?Set RUNNER_INPUT_IMAGE to a real full 2:1 panorama}"
    [[ -f "${RUNNER_INPUT_IMAGE}" ]] || { echo "Input image does not exist." >&2; exit 1; }
    mkdir -p "${DATA_DIR}/datasets/smoke"
    cp -- "${RUNNER_INPUT_IMAGE}" "${DATA_DIR}/datasets/smoke/image.png"
    run_container
    curl -fsS -X POST "http://127.0.0.1:${HOST_PORT}/run-job" \
      -H 'Content-Type: application/json' --data "@${REQUEST_FILE}"
    echo
    echo "Submitted. Use localtest.sh status or localtest.sh logs. Smoke does not wait for inference."
    ;;
  status) curl -fsS "http://127.0.0.1:${HOST_PORT}/status"; echo ;;
  logs) docker logs --tail 100 "${CONTAINER}" ;;
  down)
    mounted_data="$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/data"}}{{.Source}}{{end}}{{end}}' "${CONTAINER}")"
    owner="$(docker inspect --format '{{index .Config.Labels "deploybench.localtest"}}' "${CONTAINER}")"
    [[ "${owner}" == matrix3d ]] || { echo "Refusing to remove an unowned container." >&2; exit 1; }
    docker stop "${CONTAINER}"
    docker rm "${CONTAINER}"
    echo "Removed the test container. Inputs, weights and outputs remain in ${mounted_data:-${DATA_DIR}}."
    ;;
  *)
    echo "Usage: runner_wrapper/localtest.sh {test|build|run|smoke|status|logs|down}"
    echo "Set RUNNER_DATA_DIR to persistent storage before downloading weights."
    echo "Smoke requires RUNNER_INPUT_IMAGE. Set MATRIX3D_SMOKE_TEST=1 for the reduced 5B Turing test."
    ;;
esac
