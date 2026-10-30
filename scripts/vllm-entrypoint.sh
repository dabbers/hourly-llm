#!/usr/bin/env bash
# vLLM container entrypoint.
#
# On Docker Desktop (WSL2) hosts whose bundled NVIDIA container toolkit predates 1.14.4, Windows
# drivers 555+ fail CUDA init inside containers with "Error 500: named symbol not found" because the
# toolkit does not mount libnvdxgdmal.so.1 from the WSL driver store. compose.yaml bind-mounts
# /usr/lib/wsl/{lib,drivers}; this script puts them on the library path. On any other host the
# directories do not exist and this is a plain `vllm serve`.
set -e
if [ -d /usr/lib/wsl/lib ]; then
  nvdir=$(ls -d /usr/lib/wsl/drivers/nv_dispi*/ 2>/dev/null | head -n 1 || true)
  nvdir=${nvdir%/}
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib${nvdir:+:$nvdir}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  echo "[vllm-entrypoint] WSL2 host: LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
fi
exec vllm serve "$@"
