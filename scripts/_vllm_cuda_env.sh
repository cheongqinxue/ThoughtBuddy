# Shared CUDA env for local vLLM scripts. Source from start_* scripts.

if [ -z "${CUDA_HOME:-}" ]; then
  CUDA_HOME="$("${PYTHON}" -c "
import pathlib, sys
site = pathlib.Path(sys.prefix) / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/nvidia'
for cu_dir in sorted(site.glob('cu[0-9]*'), reverse=True):
    if (cu_dir / 'bin' / 'nvcc').is_file():
        print(cu_dir)
        break
else:
    raise SystemExit('Could not find nvcc under conda env; install cuda-nvcc or set CUDA_HOME')
")"
  export CUDA_HOME
fi
export PATH="${CUDA_HOME}/bin:${PATH}"
