#!/usr/bin/env bash
set -euo pipefail

: "${UCCL_PLUGIN_PATH:?set UCCL_PLUGIN_PATH to libnccl-net-uccl.so}"
if [[ ! -f "${UCCL_PLUGIN_PATH}" ]]; then
  echo "${UCCL_PLUGIN_PATH} does not exist" >&2
  exit 2
fi
if [[ "${UCCL_PLUGIN_PATH}" != /* ]]; then
  echo "UCCL_PLUGIN_PATH must be absolute" >&2
  exit 2
fi

export UCCL_PLUGIN_PATH
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-${UCCL_PLUGIN_PATH}}"
export BACKEND_LABEL="${BACKEND_LABEL:-nccl-uccl-plugin}"
exec bash "$(dirname "$0")/run_2node.sh"
