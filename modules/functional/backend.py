import os

from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.abspath(__file__))

# Keep compiled artifacts in a stable, project-local cache dir so the extension
# is compiled once and reused on subsequent runs (no rebuild every launch).
# Respect TORCH_EXTENSIONS_DIR if the user already set it.
_pkg_root = os.path.dirname(os.path.dirname(_src_path))  # repo root (parent of modules/)
_cache_dir = os.path.join(_pkg_root, '.torch_extensions')
if 'TORCH_EXTENSIONS_DIR' not in os.environ:
    os.environ['TORCH_EXTENSIONS_DIR'] = _cache_dir
os.makedirs(_cache_dir, exist_ok=True)

# Architecture: target the installed GPU (RTX 5060 = Blackwell sm_120).
# nvcc 12.9+ / torch cu130 both support compute_120/sm_120.
_backend = load(name='_pvcnn_backend',
                extra_cflags=['-O3', '-std=c++20'],
                extra_cuda_cflags=['-gencode=arch=compute_120,code=sm_120',
                                   '-O3',
                                   '-std=c++20'
                                   ],
                sources=[os.path.join(_src_path,'src', f) for f in [
                    'ball_query/ball_query.cpp',
                    'ball_query/ball_query.cu',
                    'grouping/grouping.cpp',
                    'grouping/grouping.cu',
                    'interpolate/neighbor_interpolate.cpp',
                    'interpolate/neighbor_interpolate.cu',
                    'interpolate/trilinear_devox.cpp',
                    'interpolate/trilinear_devox.cu',
                    'sampling/sampling.cpp',
                    'sampling/sampling.cu',
                    'voxelization/vox.cpp',
                    'voxelization/vox.cu',
                    'bindings.cpp',
                ]]
                ,verbose=True
                )

__all__ = ['_backend']
