ARCH ?= SM90
SRC := gemm_ar_compare.cu
OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 benchmark.py --preset smoke --correctness
CONFIG := pytorch

include ../../third_party/ThunderKittens/kernels/common.mk
