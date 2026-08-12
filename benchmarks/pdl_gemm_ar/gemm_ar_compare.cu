#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/torchutils.cuh"
#include "fp4_quant.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/utils/pybind.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>

// Reuse the upstream ParallelKittens fused implementation verbatim while
// suppressing its Python module definition.  This keeps the fused control tied
// to the pinned ThunderKittens submodule instead of maintaining a copied body.
#define config pk_fused_config
#define globals pk_fused_globals
#define lcsct pk_fused_lcsct
#define entrypoint pk_fused_entrypoint
#define epilogue_kernel pk_fused_epilogue_kernel
#pragma push_macro("PYBIND11_MODULE")
#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, variable) \
    static void ignored_upstream_module(pybind11::module_ &variable)
#include "../../third_party/ThunderKittens/kernels/parallel/gemm_ar/gemm_ar_h100_lcsc.cu"
#pragma pop_macro("PYBIND11_MODULE")
#undef epilogue_kernel
#undef entrypoint
#undef lcsct
#undef globals
#undef config

using namespace kittens;

namespace pdl_gemm_ar {

constexpr int NUM_SMS = pk_fused_config::NUM_BLOCKS;
constexpr int TRACE_POINTS = 4;
constexpr uint32_t MAX_HIERARCHICAL_EPOCH = (1u << 31) - 1;

__device__ __forceinline__ coord<ducks::default_type> grid_local_count() {
    return {1, 0, 1};
}

__device__ __forceinline__ coord<ducks::default_type> grid_ready_count() {
    return {1, 0, 2};
}

__device__ __forceinline__ coord<ducks::default_type> reset_pre_count() {
    return {1, 0, 3};
}

__device__ __forceinline__ coord<ducks::default_type> reset_post_count() {
    return {1, 0, 4};
}

__device__ __forceinline__ coord<ducks::default_type>
hierarchical_join_slot(int source_dev_idx) {
    return {1, 0, 5 + source_dev_idx};
}

template <typename Globals>
__device__ __forceinline__ int grid_arrive_acq_rel(
    const Globals &G
) {
    int previous;
    auto *counter = &G.barrier[G.dev_idx][grid_local_count()];
    asm volatile(
        "atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;"
        : "=r"(previous)
        : "l"(counter)
        : "memory"
    );
    return previous;
}

enum class WaitMode : int {
    NONE = 0,
    TILE_COUNTER = 1,
    GRID_COUNTER = 2,
    PDL_GRID = 3,
    TILE_COUNTER_PER_RANK = 4,
};

enum class LaunchMode : int {
    DEFAULT_STREAM = 0,
    PDL_GRID = 1,
    PDL_TILE = 2,
    TWO_STREAM = 3,
};

enum class Fp4Mode : int {
    PRE = 0,
    POST = 1,
};

struct fp4_globals {
    static constexpr int NUM_DEVICES = pk_fused_globals::NUM_DEVICES;
    static constexpr int PIPELINE_STAGES = pk_fused_globals::PIPELINE_STAGES;
    static constexpr int SUPER_M = pk_fused_globals::SUPER_M;
    static constexpr int ROW_BLOCK = pk_fused_globals::ROW_BLOCK;
    static constexpr int COL_BLOCK = pk_fused_globals::COL_BLOCK;
    static constexpr int RED_BLOCK = pk_fused_globals::RED_BLOCK;

    using A_tile = pk_fused_globals::A_tile;
    using B_tile = pk_fused_globals::B_tile;
    using C_tile = pk_fused_globals::C_tile;
    using A_gl = pk_fused_globals::A_gl;
    using B_gl = pk_fused_globals::B_gl;
    using C_pgl = pk_fused_globals::C_pgl;
    using barrier_pgl = pk_fused_globals::barrier_pgl;
    using data_gl = gl<uint8, 1, NUM_DEVICES, -1, -1>;
    using data_pgl = pgl<data_gl, NUM_DEVICES, true>;
    using scale_gl = gl<bf16, 1, NUM_DEVICES, -1, -1>;
    using scale_pgl = pgl<scale_gl, NUM_DEVICES, true>;

    A_gl A;
    B_gl B;
    C_pgl C;
    data_pgl data;
    scale_pgl scales;
    barrier_pgl barrier;
    const int dev_idx;
    const int num_comm_sms;
    const int num_comp_sms;
};

struct fp4_pipeline_inputs {
    fp4_globals::A_tile A[2];
    fp4_globals::B_tile B;
};

// Four input stages are needed after the output tile is removed from the
// overlaid pipeline allocation.  Keep a little alignment slack for the TMA
// allocator while retaining the original static semaphore reservation.
constexpr int FP4_DYNAMIC_SHARED_MEMORY =
    ((sizeof(fp4_pipeline_inputs) * fp4_globals::PIPELINE_STAGES + 1023) / 1024)
    * 1024 + 1024;

__device__ __forceinline__ uint64_t globaltimer() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

template <bool INSTRUMENT>
__device__ __forceinline__ void trace_timestamp(
    int64_t *trace,
    int cta,
    int point
) {
    if constexpr (INSTRUMENT) {
        trace[static_cast<size_t>(cta) * TRACE_POINTS + point] =
            static_cast<int64_t>(globaltimer());
    }
}

template <bool INSTRUMENT>
__device__ __forceinline__ void smid_write(int *smids, int cta) {
    if constexpr (INSTRUMENT) smids[cta] = kittens::smid();
}

__device__ __forceinline__ void task_to_tile(
    int task_id,
    int super_rows,
    int final_rows,
    int super_blocks,
    int col_blocks,
    int &row_idx,
    int &col_idx
) {
    if (task_id < super_rows * col_blocks) {
        row_idx = pk_fused_globals::SUPER_M * (task_id / super_blocks)
            + task_id % pk_fused_globals::SUPER_M;
        col_idx = (task_id % super_blocks) / pk_fused_globals::SUPER_M;
    } else {
        int remainder_id = task_id - super_rows * col_blocks;
        row_idx = super_rows + remainder_id % final_rows;
        col_idx = remainder_id / final_rows;
    }
}

template <int NUM_DEVICES>
__device__ __forceinline__ void wait_counter_acquire(
    const barrier_t<NUM_DEVICES> &barrier,
    const coord<ducks::default_type> &idx,
    int dev_idx,
    int expected
) {
    int value;
    do {
        asm volatile(
            "ld.relaxed.sys.global.s32 %0, [%1];"
            : "=r"(value)
            : "l"(&barrier[dev_idx][idx])
            : "memory"
        );
    } while (value != expected);
    asm volatile(
        "ld.acquire.sys.global.s32 %0, [%1];"
        : "=r"(value)
        : "l"(&barrier[dev_idx][idx])
        : "memory"
    );
}

__device__ __forceinline__ void quantize_accumulator_to_local(
    const fp4_globals &G,
    const rt_fl<
        pk_fused_globals::ROW_BLOCK / 8,
        pk_fused_globals::COL_BLOCK
    > &accum,
    int row_idx,
    int col_idx,
    int warpgroup_id
) {
    const int lane = warp::laneid();
    const int lane_in_row = lane & 3;
    const int warp_in_warpgroup = warpgroup::warpid();
    const int global_row0 =
        row_idx * fp4_globals::ROW_BLOCK
        + warpgroup_id * (fp4_globals::ROW_BLOCK / 2)
        + warp_in_warpgroup * 16
        + lane / 4;
    const int global_row1 = global_row0 + 8;
    const int data_col_base =
        col_idx * (fp4_globals::COL_BLOCK / 2);
    const int scale_col_base =
        col_idx * (fp4_globals::COL_BLOCK / 16);

    #pragma unroll
    for (int pair = 0; pair < fp4_globals::COL_BLOCK / 32; ++pair) {
        const int j0 = pair * 2;
        const int j1 = j0 + 1;
        // The FP32 WGMMA fragment is already in the logical row-major
        // register layout.  The lane-to-column map is the same map used by
        // the regular BF16 epilogue: lanes 0..3 own the two values at
        // columns 0/1, 2/3, 8/9, and 10/11 of the corresponding 16x16
        // subtile.  The x/y swap in shared_to_register::store is only needed
        // when materializing the swizzled SMEM representation; applying it
        // here would change the GEMM result's logical columns.
        float2 reg_tmp0[4];
        float2 reg_tmp1[4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            reg_tmp0[k] = accum.tiles[0][j0].data[k];
            reg_tmp1[k] = accum.tiles[0][j1].data[k];
        }

        const float row00_values[4] = {
            fp4::rounded_bf16(reg_tmp0[0].x),
            fp4::rounded_bf16(reg_tmp0[0].y),
            fp4::rounded_bf16(reg_tmp0[2].x),
            fp4::rounded_bf16(reg_tmp0[2].y),
        };
        const float row01_values[4] = {
            fp4::rounded_bf16(reg_tmp1[0].x),
            fp4::rounded_bf16(reg_tmp1[0].y),
            fp4::rounded_bf16(reg_tmp1[2].x),
            fp4::rounded_bf16(reg_tmp1[2].y),
        };
        const float row10_values[4] = {
            fp4::rounded_bf16(reg_tmp0[1].x),
            fp4::rounded_bf16(reg_tmp0[1].y),
            fp4::rounded_bf16(reg_tmp0[3].x),
            fp4::rounded_bf16(reg_tmp0[3].y),
        };
        const float row11_values[4] = {
            fp4::rounded_bf16(reg_tmp1[1].x),
            fp4::rounded_bf16(reg_tmp1[1].y),
            fp4::rounded_bf16(reg_tmp1[3].x),
            fp4::rounded_bf16(reg_tmp1[3].y),
        };
        float amax00 = 0.0f;
        float amax01 = 0.0f;
        float amax10 = 0.0f;
        float amax11 = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            amax00 = fmaxf(amax00, fabsf(row00_values[i]));
            amax01 = fmaxf(amax01, fabsf(row01_values[i]));
            amax10 = fmaxf(amax10, fabsf(row10_values[i]));
            amax11 = fmaxf(amax11, fabsf(row11_values[i]));
        }
        amax00 = fp4::subgroup_max<4>(amax00);
        amax01 = fp4::subgroup_max<4>(amax01);
        amax10 = fp4::subgroup_max<4>(amax10);
        amax11 = fp4::subgroup_max<4>(amax11);
        const bf16 scale00 = fp4::decode_scale(amax00);
        const bf16 scale01 = fp4::decode_scale(amax01);
        const bf16 scale10 = fp4::decode_scale(amax10);
        const bf16 scale11 = fp4::decode_scale(amax11);
        const float scale00_float = fp4::scale_as_float(scale00);
        const float scale01_float = fp4::scale_as_float(scale01);
        const float scale10_float = fp4::scale_as_float(scale10);
        const float scale11_float = fp4::scale_as_float(scale11);

        // Each lane owns two logical FP4 values in the 16-column scale
        // block.  Four lanes therefore cover eight bytes; the second pair
        // is four bytes farther in the packed row.
        const int byte_col = data_col_base + pair * 16 + lane_in_row;
        const uint8_t data000 = fp4::encode_pair(
            fp4::normalize(row00_values[0], scale00_float),
            fp4::normalize(row00_values[1], scale00_float)
        );
        const uint8_t data001 = fp4::encode_pair(
            fp4::normalize(row00_values[2], scale00_float),
            fp4::normalize(row00_values[3], scale00_float)
        );
        const uint8_t data010 = fp4::encode_pair(
            fp4::normalize(row01_values[0], scale01_float),
            fp4::normalize(row01_values[1], scale01_float)
        );
        const uint8_t data011 = fp4::encode_pair(
            fp4::normalize(row01_values[2], scale01_float),
            fp4::normalize(row01_values[3], scale01_float)
        );
        const uint8_t data100 = fp4::encode_pair(
            fp4::normalize(row10_values[0], scale10_float),
            fp4::normalize(row10_values[1], scale10_float)
        );
        const uint8_t data101 = fp4::encode_pair(
            fp4::normalize(row10_values[2], scale10_float),
            fp4::normalize(row10_values[3], scale10_float)
        );
        const uint8_t data110 = fp4::encode_pair(
            fp4::normalize(row11_values[0], scale11_float),
            fp4::normalize(row11_values[1], scale11_float)
        );
        const uint8_t data111 = fp4::encode_pair(
            fp4::normalize(row11_values[2], scale11_float),
            fp4::normalize(row11_values[3], scale11_float)
        );
        const uint32_t data_word000 = fp4::gather_u8x4(data000, lane);
        const uint32_t data_word001 = fp4::gather_u8x4(data001, lane);
        const uint32_t data_word010 = fp4::gather_u8x4(data010, lane);
        const uint32_t data_word011 = fp4::gather_u8x4(data011, lane);
        const uint32_t data_word100 = fp4::gather_u8x4(data100, lane);
        const uint32_t data_word101 = fp4::gather_u8x4(data101, lane);
        const uint32_t data_word110 = fp4::gather_u8x4(data110, lane);
        const uint32_t data_word111 = fp4::gather_u8x4(data111, lane);
        if (lane_in_row == 0) {
            fp4::peer_store_u32(
                reinterpret_cast<uint32_t *>(&G.data[G.dev_idx][{
                    0, G.dev_idx, global_row0, byte_col
                }]),
                data_word000
            );
            fp4::peer_store_u32(
                reinterpret_cast<uint32_t *>(&G.data[G.dev_idx][{
                    0, G.dev_idx, global_row0, byte_col + 4
                }]),
                data_word001
            );
            fp4::peer_store_u32(
                reinterpret_cast<uint32_t *>(&G.data[G.dev_idx][{
                    0, G.dev_idx, global_row0, byte_col + 8
                }]),
                data_word010
            );
            fp4::peer_store_u32(
                reinterpret_cast<uint32_t *>(&G.data[G.dev_idx][{
                    0, G.dev_idx, global_row0, byte_col + 12
                }]),
                data_word011
            );
            fp4::peer_store_u32(
                reinterpret_cast<uint32_t *>(&G.data[G.dev_idx][{
                    0, G.dev_idx, global_row1, byte_col
                }]),
                data_word100
            );
            fp4::peer_store_u32(
                reinterpret_cast<uint32_t *>(&G.data[G.dev_idx][{
                    0, G.dev_idx, global_row1, byte_col + 4
                }]),
                data_word101
            );
            fp4::peer_store_u32(
                reinterpret_cast<uint32_t *>(&G.data[G.dev_idx][{
                    0, G.dev_idx, global_row1, byte_col + 8
                }]),
                data_word110
            );
            fp4::peer_store_u32(
                reinterpret_cast<uint32_t *>(&G.data[G.dev_idx][{
                    0, G.dev_idx, global_row1, byte_col + 12
                }]),
                data_word111
            );
            fp4::peer_store_u32(
                reinterpret_cast<uint32_t *>(&G.scales[G.dev_idx][{
                    0, G.dev_idx, global_row0, scale_col_base + j0
                }]),
                fp4::pack_bf16x2_bits(scale00, scale01)
            );
            fp4::peer_store_u32(
                reinterpret_cast<uint32_t *>(&G.scales[G.dev_idx][{
                    0, G.dev_idx, global_row1, scale_col_base + j0
                }]),
                fp4::pack_bf16x2_bits(scale10, scale11)
            );
        }
    }

}

template <bool INSTRUMENT, bool DIRECT_FP4, typename Globals>
__device__ inline void split_compute_body(
    const Globals &G,
    bool trigger_pdl,
    bool publish_grid_ready,
    int64_t *trace,
    int *smids
) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator(reinterpret_cast<int *>(&__shm[0]));

    using pipeline_inputs = fp4_pipeline_inputs;
    struct pipeline_outputs {
        pk_fused_globals::C_tile C[2];
    };

    if constexpr (DIRECT_FP4) {
        static_assert(
            sizeof(pipeline_inputs) * pk_fused_globals::PIPELINE_STAGES
                <= FP4_DYNAMIC_SHARED_MEMORY
        );
    } else {
        static_assert(
            sizeof(pipeline_inputs) * (pk_fused_globals::PIPELINE_STAGES - 1)
                    + sizeof(pipeline_outputs)
                <= pk_fused_config::DYNAMIC_SHARED_MEMORY
        );
    }

    pipeline_inputs *inputs = nullptr;
    pipeline_outputs *outputs = nullptr;
    if constexpr (DIRECT_FP4) {
        auto &input_storage = allocator.allocate<
            pipeline_inputs, pk_fused_globals::PIPELINE_STAGES
        >();
        inputs = &input_storage[0];
    } else {
        struct split_storage {
            pipeline_inputs inputs[pk_fused_globals::PIPELINE_STAGES - 1];
            pipeline_outputs outputs;
        };
        auto &storage = allocator.allocate<split_storage>();
        inputs = &storage.inputs[0];
        outputs = &storage.outputs;
    }

    __shared__ semaphore inputs_arrived[pk_fused_globals::PIPELINE_STAGES];
    __shared__ semaphore inputs_finished[pk_fused_globals::PIPELINE_STAGES];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;

    if (threadIdx.x == 0) {
        trace_timestamp<INSTRUMENT>(trace, blockIdx.x, 0);
        smid_write<INSTRUMENT>(smids, blockIdx.x);
        #pragma unroll
        for (int i = 0; i < pk_fused_globals::PIPELINE_STAGES; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 8);
        }
        init_semaphore(outputs_arrived, 0, 2);
        init_semaphore(outputs_finished, 0, 1);
    }
    __syncthreads();

    int warpgroup_id = warpgroup::groupid();
    int warp_id = warpgroup::warpid();
    int lane_id = warp::laneid();
    int stage = 0;
    uint32_t phasebits = 0xFFFF0000;
    int row_blocks = G.A.rows() / pk_fused_globals::ROW_BLOCK;
    int col_blocks = G.B.cols() / pk_fused_globals::COL_BLOCK;
    int super_rows = (row_blocks / pk_fused_globals::SUPER_M)
        * pk_fused_globals::SUPER_M;
    int final_rows = row_blocks - super_rows;
    int super_blocks = pk_fused_globals::SUPER_M * col_blocks;
    int num_blocks = row_blocks * col_blocks;
    int num_iters = G.A.cols() / pk_fused_globals::RED_BLOCK;

    if (warpgroup_id == pk_fused_config::NUM_WARPGROUPS - 1) {
        warpgroup::decrease_registers<pk_fused_config::PRODUCER_REGISTERS>();

        if (warp_id == 0 && lane_id == 0) {
            bool first_load_issued = false;
            for (int task_id = blockIdx.x; task_id < num_blocks;
                 task_id += G.num_comp_sms) {
                int row_idx, col_idx;
                task_to_tile(
                    task_id, super_rows, final_rows, super_blocks, col_blocks,
                    row_idx, col_idx
                );

                for (int red_idx = 0; red_idx < num_iters; ++red_idx) {
                    wait(
                        inputs_finished[stage],
                        get_phasebit<1>(phasebits, stage)
                    );
                    update_phasebit<1>(phasebits, stage);
                    tma::expect_bytes(
                        inputs_arrived[stage], sizeof(pipeline_inputs)
                    );
                    if (
                        !DIRECT_FP4
                        && red_idx == pk_fused_globals::PIPELINE_STAGES - 1
                    ) {
                        wait(
                            outputs_finished,
                            get_phasebit<1>(
                                phasebits, pk_fused_globals::PIPELINE_STAGES
                            )
                        );
                        update_phasebit<1>(
                            phasebits, pk_fused_globals::PIPELINE_STAGES
                        );
                    }
                    #pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        tma::load_async(
                            inputs[stage].A[i], G.A,
                            {row_idx * 2 + i, red_idx}, inputs_arrived[stage]
                        );
                    }
                    tma::load_async(
                        inputs[stage].B, G.B, {red_idx, col_idx},
                        inputs_arrived[stage]
                    );

                    // Release the programmatic launch port only after this CTA
                    // has issued useful independent input traffic.  Tile data
                    // visibility is still carried by the system-scope counter.
                    if (!first_load_issued) {
                        if (trigger_pdl) kittens::pdl::arrive();
                        trace_timestamp<INSTRUMENT>(
                            trace, blockIdx.x, 1
                        );
                        first_load_issued = true;
                    }
                    stage = (stage + 1) % pk_fused_globals::PIPELINE_STAGES;
                }
            }
        } else if (warp_id == 1 && lane_id == 0) {
            bool first_store_completed = false;
            for (int task_id = blockIdx.x; task_id < num_blocks;
                 task_id += G.num_comp_sms) {
                int row_idx, col_idx;
                task_to_tile(
                    task_id, super_rows, final_rows, super_blocks, col_blocks,
                    row_idx, col_idx
                );

                wait(outputs_arrived, get_phasebit<0>(phasebits, 0));
                update_phasebit<0>(phasebits, 0);
                if constexpr (!DIRECT_FP4) {
                    #pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        tma::store_async(
                            G.C[G.dev_idx], outputs->C[i],
                            {row_idx * 2 + i, col_idx}
                        );
                    }
                    tma::store_async_read_wait();
                    arrive(outputs_finished);
                }

                int signal_dev_idx = task_id % pk_fused_globals::NUM_DEVICES;
                signal(G.barrier, {row_idx, col_idx}, signal_dev_idx, 1);
                // DIRECT_FP4 has no output-SMEM lifetime to protect, but it
                // still needs a phase-lifetime handshake.  Without this ack,
                // the consumer warpgroups can complete two later tiles before
                // this warp observes outputs_arrived.  The two-phase mbarrier
                // then wraps back to the parity being waited on and the CTA
                // can spin forever.  In the SMEM path the same semaphore is
                // already the output-buffer reuse acknowledgement.
                if constexpr (DIRECT_FP4) {
                    arrive(outputs_finished);
                }
                if (!first_store_completed) {
                    trace_timestamp<INSTRUMENT>(
                        trace, blockIdx.x, 2
                    );
                    first_store_completed = true;
                }
            }
        }
    } else {
        warpgroup::increase_registers<pk_fused_config::CONSUMER_REGISTERS>();
        bool direct_output_pending = false;

        for (int task_id = blockIdx.x; task_id < num_blocks;
             task_id += G.num_comp_sms) {
            rt_fl<
                pk_fused_globals::ROW_BLOCK / 8,
                pk_fused_globals::COL_BLOCK
            > C_accum;
            warp::zero(C_accum);

            for (int red_idx = 0; red_idx < num_iters; ++red_idx) {
                wait(
                    inputs_arrived[stage], get_phasebit<0>(phasebits, stage)
                );
                update_phasebit<0>(phasebits, stage);
                warpgroup::mma_AB(
                    C_accum, inputs[stage].A[warpgroup_id], inputs[stage].B
                );
                warpgroup::mma_async_wait();
                warp::arrive(inputs_finished[stage]);
                stage = (stage + 1) % pk_fused_globals::PIPELINE_STAGES;
            }

            group<8>::sync(3);
            if constexpr (DIRECT_FP4) {
                int row_idx, col_idx;
                task_to_tile(
                    task_id, super_rows, final_rows, super_blocks, col_blocks,
                    row_idx, col_idx
                );
                quantize_accumulator_to_local(
                    G, C_accum, row_idx, col_idx, warpgroup_id
                );
                // PRE writes only the current rank's local wire buffer.  The
                // communication grid reads all eight peer allocations after
                // the tile-ready system counter, avoiding multicast/unicast
                // virtual aliases and eightfold payload replication.
                warpgroup::sync(warpgroup_id + 1);
            } else {
                warpgroup::store(outputs->C[warpgroup_id], C_accum);
                warpgroup::sync(warpgroup_id + 1);
            }
            if constexpr (DIRECT_FP4) {
                // One tile of direct-output lookahead is safe: while the
                // storer publishes tile t, consumers may compute and quantize
                // tile t+1 into its disjoint global destination.  Before they
                // flip outputs_arrived for t+1, however, they must know the
                // storer observed t.  Otherwise two quick phase flips can lap
                // the observer and make its parity wait permanent.
                if (direct_output_pending) {
                    warpgroup::wait(
                        outputs_finished,
                        get_phasebit<0>(
                            phasebits, pk_fused_globals::PIPELINE_STAGES
                        )
                    );
                    update_phasebit<0>(
                        phasebits, pk_fused_globals::PIPELINE_STAGES
                    );
                }
                direct_output_pending = true;
            }
            warpgroup::arrive(outputs_arrived);
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        if (trigger_pdl) kittens::pdl::arrive();
        if (publish_grid_ready) {
            int previous = grid_arrive_acq_rel(G);
            if (previous == gridDim.x - 1) {
                signal_all(G.barrier, grid_ready_count(), 1);
            }
        }
        trace_timestamp<INSTRUMENT>(trace, blockIdx.x, 3);
    }
}

template <bool INSTRUMENT>
__global__ __launch_bounds__(pk_fused_config::NUM_THREADS, 1)
void split_compute_kernel(
    const __grid_constant__ pk_fused_globals G,
    bool trigger_pdl,
    bool publish_grid_ready,
    int64_t *trace,
    int *smids
) {
    split_compute_body<INSTRUMENT, false>(
        G, trigger_pdl, publish_grid_ready, trace, smids
    );
}

template <bool INSTRUMENT>
__global__ __launch_bounds__(pk_fused_config::NUM_THREADS, 1)
void fp4_pre_compute_kernel(
    const __grid_constant__ fp4_globals G,
    bool trigger_pdl,
    bool publish_grid_ready,
    int64_t *trace,
    int *smids
) {
    split_compute_body<INSTRUMENT, true>(
        G, trigger_pdl, publish_grid_ready, trace, smids
    );
}

template <bool INSTRUMENT>
__global__ __launch_bounds__(pk_fused_config::NUM_THREADS, 1)
void fp4_post_compute_kernel(
    const __grid_constant__ fp4_globals G,
    bool trigger_pdl,
    bool publish_grid_ready,
    int64_t *trace,
    int *smids
) {
    split_compute_body<INSTRUMENT, false>(
        G, trigger_pdl, publish_grid_ready, trace, smids
    );
}

void launch_fp4_post_compute(
    const fp4_globals &G,
    int num_ctas,
    bool trigger_pdl,
    bool publish_grid_ready,
    bool instrument,
    cudaStream_t stream,
    int64_t *trace,
    int *smids
) {
    if (instrument) {
        auto kernel = fp4_post_compute_kernel<true>;
        CUDACHECK(cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            pk_fused_config::DYNAMIC_SHARED_MEMORY
        ));
        kernel<<<
            num_ctas, pk_fused_config::NUM_THREADS,
            pk_fused_config::DYNAMIC_SHARED_MEMORY, stream
        >>>(G, trigger_pdl, publish_grid_ready, trace, smids);
    } else {
        auto kernel = fp4_post_compute_kernel<false>;
        CUDACHECK(cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            pk_fused_config::DYNAMIC_SHARED_MEMORY
        ));
        kernel<<<
            num_ctas, pk_fused_config::NUM_THREADS,
            pk_fused_config::DYNAMIC_SHARED_MEMORY, stream
        >>>(G, trigger_pdl, publish_grid_ready, nullptr, nullptr);
    }
    CUDACHECK(cudaGetLastError());
}

template <int NUM_THREADS, bool INSTRUMENT>
__global__ __launch_bounds__(NUM_THREADS, 1)
void split_communication_kernel(
    const __grid_constant__ pk_fused_globals G,
    WaitMode wait_mode,
    uint32_t *resident_counter,
    int64_t *trace,
    int *smids
) {
    constexpr int NUM_WARPS = NUM_THREADS / WARP_THREADS;
    static_assert(NUM_THREADS % WARP_THREADS == 0);

    if (threadIdx.x == 0) {
        trace_timestamp<INSTRUMENT>(trace, blockIdx.x, 0);
        smid_write<INSTRUMENT>(smids, blockIdx.x);
        if (resident_counter != nullptr) {
            atomicAdd(resident_counter, 1u);
        }
    }

    int row_blocks = G.A.rows() / pk_fused_globals::ROW_BLOCK;
    int col_blocks = G.B.cols() / pk_fused_globals::COL_BLOCK;
    int super_rows = (row_blocks / pk_fused_globals::SUPER_M)
        * pk_fused_globals::SUPER_M;
    int final_rows = row_blocks - super_rows;
    int super_blocks = pk_fused_globals::SUPER_M * col_blocks;
    int num_blocks = row_blocks * col_blocks;
    bool first_task = true;

    if (wait_mode == WaitMode::PDL_GRID) {
        kittens::pdl::wait();
    }
    if (
        wait_mode == WaitMode::GRID_COUNTER
        || wait_mode == WaitMode::PDL_GRID
    ) {
        if (threadIdx.x == 0) {
            wait_counter_acquire(
                G.barrier, grid_ready_count(), G.dev_idx,
                pk_fused_globals::NUM_DEVICES
            );
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        trace_timestamp<INSTRUMENT>(trace, blockIdx.x, 1);
    }

    for (int task_id = pk_fused_globals::NUM_DEVICES * blockIdx.x + G.dev_idx;
         task_id < num_blocks;
         task_id += pk_fused_globals::NUM_DEVICES * gridDim.x) {
        int row_idx, col_idx;
        task_to_tile(
            task_id, super_rows, final_rows, super_blocks, col_blocks,
            row_idx, col_idx
        );

        if (wait_mode == WaitMode::TILE_COUNTER) {
            if (threadIdx.x == 0) {
                wait_counter_acquire(
                    G.barrier, {row_idx, col_idx}, G.dev_idx,
                    pk_fused_globals::NUM_DEVICES
                );
            }
            __syncthreads();
        }
        if (first_task && threadIdx.x == 0) {
            trace_timestamp<INSTRUMENT>(trace, blockIdx.x, 2);
        }

        constexpr int WARPS_PER_ROW =
            pk_fused_globals::COL_BLOCK / (WARP_THREADS * 2);
        constexpr int WORK_ITEMS =
            pk_fused_globals::ROW_BLOCK * WARPS_PER_ROW;
        int lane = threadIdx.x % WARP_THREADS;
        for (int i = threadIdx.x / WARP_THREADS;
             i < WORK_ITEMS;
             i += NUM_WARPS) {
            int global_row =
                row_idx * pk_fused_globals::ROW_BLOCK
                + i / WARPS_PER_ROW;
            int global_col =
                col_idx * pk_fused_globals::COL_BLOCK
                + (i % WARPS_PER_ROW) * WARP_THREADS * 2
                + lane * 2;
            bf16_2 value;
            auto *multicast_ptr = reinterpret_cast<bf16_2 *>(
                G.C.mc_ptr_at(
                    coord<ducks::default_type>(
                        0, 0, global_row, global_col
                    )
                )
            );
            multimem<bf16_2>::ld_reduce<reduce_op::ADD>(
                value, multicast_ptr
            );
            multimem<bf16_2>::st(multicast_ptr, value);
        }
        first_task = false;
    }

    if (threadIdx.x == 0) {
        trace_timestamp<INSTRUMENT>(trace, blockIdx.x, 3);
    }
}

constexpr int HIERARCHICAL_TILE_ELEMENTS =
    pk_fused_globals::ROW_BLOCK * pk_fused_globals::COL_BLOCK;

__device__ __forceinline__ void publish_owner_slot(
    uint32_t *ready,
    uint32_t *error,
    int slot,
    uint32_t epoch
) {
    uint32_t previous;
    asm volatile(
        "atom.acq_rel.sys.global.cas.b32 %0, [%1], %2, %3;"
        : "=r"(previous)
        : "l"(&ready[slot]), "r"(epoch - 1), "r"(epoch)
        : "memory"
    );
    if (previous != epoch - 1) {
        uint32_t ignored;
        const uint32_t code = 0x40000000u | static_cast<uint32_t>(slot);
        asm volatile(
            "atom.relaxed.sys.global.cas.b32 %0, [%1], %2, %3;"
            : "=r"(ignored)
            : "l"(error), "r"(0u), "r"(code)
            : "memory"
        );
        asm volatile(
            "st.release.sys.global.u32 [%0], %1;"
            :: "l"(&ready[slot]), "r"(epoch) : "memory"
        );
    }
}

template <int NUM_THREADS>
__global__ __launch_bounds__(NUM_THREADS, 1)
void hierarchical_pack_kernel(
    const __grid_constant__ pk_fused_globals G,
    bf16 *wire,
    uint32_t *ready,
    uint32_t *error,
    uint32_t epoch,
    WaitMode wait_mode
) {
    const int row_blocks = G.A.rows() / pk_fused_globals::ROW_BLOCK;
    const int col_blocks = G.B.cols() / pk_fused_globals::COL_BLOCK;
    const int super_rows = (row_blocks / pk_fused_globals::SUPER_M)
        * pk_fused_globals::SUPER_M;
    const int final_rows = row_blocks - super_rows;
    const int super_blocks = pk_fused_globals::SUPER_M * col_blocks;
    const int num_blocks = row_blocks * col_blocks;

    if (wait_mode == WaitMode::PDL_GRID) kittens::pdl::wait();
    if (wait_mode == WaitMode::GRID_COUNTER || wait_mode == WaitMode::PDL_GRID) {
        if (threadIdx.x == 0) {
            wait_counter_acquire(
                G.barrier,
                grid_ready_count(),
                G.dev_idx,
                pk_fused_globals::NUM_DEVICES
            );
        }
        __syncthreads();
    }

    for (
        int task_id = pk_fused_globals::NUM_DEVICES * blockIdx.x + G.dev_idx;
        task_id < num_blocks;
        task_id += pk_fused_globals::NUM_DEVICES * gridDim.x
    ) {
        int row_idx, col_idx;
        task_to_tile(
            task_id,
            super_rows,
            final_rows,
            super_blocks,
            col_blocks,
            row_idx,
            col_idx
        );
        if (wait_mode == WaitMode::TILE_COUNTER) {
            if (threadIdx.x == 0) {
                wait_counter_acquire(
                    G.barrier,
                    {row_idx, col_idx},
                    G.dev_idx,
                    pk_fused_globals::NUM_DEVICES
                );
            }
            __syncthreads();
        }

        const int slot = task_id / pk_fused_globals::NUM_DEVICES;
        auto *wire_pairs = reinterpret_cast<bf16_2 *>(
            wire + static_cast<size_t>(slot) * HIERARCHICAL_TILE_ELEMENTS
        );
        constexpr int PAIRS_PER_TILE = HIERARCHICAL_TILE_ELEMENTS / 2;
        for (int pair = threadIdx.x; pair < PAIRS_PER_TILE; pair += blockDim.x) {
            const int element = pair * 2;
            const int local_row = element / pk_fused_globals::COL_BLOCK;
            const int local_col = element % pk_fused_globals::COL_BLOCK;
            auto *multicast_ptr = reinterpret_cast<bf16_2 *>(
                G.C.mc_ptr_at({
                    0,
                    0,
                    row_idx * pk_fused_globals::ROW_BLOCK + local_row,
                    col_idx * pk_fused_globals::COL_BLOCK + local_col
                })
            );
            bf16_2 reduced;
            multimem<bf16_2>::ld_reduce<
                reduce_op::ADD,
                memory_model::STRONG
            >(reduced, multicast_ptr);
            move<bf16_2>::stg(&wire_pairs[pair], reduced);
        }
        // Every writer publishes its own stores before thread 0 advances the
        // slot epoch.  The network stream waits on that epoch before NCCL or
        // a future RDMA backend reads the contiguous owner wire.
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            publish_owner_slot(ready, error, slot, epoch);
        }
        __syncthreads();
    }
}

template <int NUM_THREADS>
__global__ __launch_bounds__(NUM_THREADS, 1)
void hierarchical_unpack_kernel(
    pk_fused_globals::C_pgl C,
    const bf16 *wire,
    int dev_idx,
    int row_blocks,
    int col_blocks,
    int first_slot,
    int slot_count
) {
    const int slot = first_slot + blockIdx.x;
    if (blockIdx.x >= slot_count) return;
    const int task_id = slot * pk_fused_globals::NUM_DEVICES + dev_idx;
    const int num_blocks = row_blocks * col_blocks;
    if (task_id >= num_blocks) return;

    const int super_rows = (row_blocks / pk_fused_globals::SUPER_M)
        * pk_fused_globals::SUPER_M;
    const int final_rows = row_blocks - super_rows;
    const int super_blocks = pk_fused_globals::SUPER_M * col_blocks;
    int row_idx, col_idx;
    task_to_tile(
        task_id,
        super_rows,
        final_rows,
        super_blocks,
        col_blocks,
        row_idx,
        col_idx
    );

    const auto *wire_pairs = reinterpret_cast<const bf16_2 *>(
        wire + static_cast<size_t>(slot) * HIERARCHICAL_TILE_ELEMENTS
    );
    constexpr int PAIRS_PER_TILE = HIERARCHICAL_TILE_ELEMENTS / 2;
    for (int pair = threadIdx.x; pair < PAIRS_PER_TILE; pair += blockDim.x) {
        const int element = pair * 2;
        const int local_row = element / pk_fused_globals::COL_BLOCK;
        const int local_col = element % pk_fused_globals::COL_BLOCK;
        bf16_2 value;
        asm volatile(
            "ld.acquire.sys.global.u32 %0, [%1];"
            : "=r"(*reinterpret_cast<uint32_t *>(&value))
            : "l"(&wire_pairs[pair])
            : "memory"
        );
        auto *multicast_ptr = reinterpret_cast<bf16_2 *>(
            C.mc_ptr_at({
                0,
                0,
                row_idx * pk_fused_globals::ROW_BLOCK + local_row,
                col_idx * pk_fused_globals::COL_BLOCK + local_col
            })
        );
        multimem<bf16_2>::st<memory_model::STRONG>(multicast_ptr, value);
    }
}

__global__ void hierarchical_wait_ready_kernel(
    const uint32_t *ready,
    uint32_t *error,
    int first_slot,
    int slot_count,
    uint32_t epoch,
    uint64_t timeout_ns
) {
    if (threadIdx.x >= slot_count) return;
    const int slot = first_slot + threadIdx.x;
    const uint64_t start = globaltimer();
    uint32_t value;
    do {
        asm volatile(
            "ld.acquire.sys.global.u32 %0, [%1];"
            : "=r"(value)
            : "l"(&ready[slot])
            : "memory"
        );
        if (value == epoch) return;
        if (globaltimer() - start >= timeout_ns) {
            uint32_t ignored;
            const uint32_t code =
                0x20000000u | static_cast<uint32_t>(slot);
            asm volatile(
                "atom.relaxed.sys.global.cas.b32 %0, [%1], %2, %3;"
                : "=r"(ignored)
                : "l"(error), "r"(0u), "r"(code)
                : "memory"
            );
            return;
        }
    } while (true);
}

__global__ void hierarchical_local_join_kernel(
    pk_fused_globals::barrier_pgl barrier,
    uint32_t *error,
    int dev_idx,
    uint64_t timeout_ns
) {
    if (threadIdx.x != 0) return;
    // Each owner rank reaches this kernel after its multicast unpack stream.
    // Every rank is the sole writer of one monotonic sequence slot.  Separate
    // slots give the waiter a direct acquire from every rank's release and
    // remain reusable even if a fast rank publishes the following round.
    const auto own_slot = hierarchical_join_slot(dev_idx);
    uint32_t before;
    asm volatile(
        "ld.relaxed.sys.global.u32 %0, [%1];"
        : "=r"(before)
        : "l"(&barrier[dev_idx][own_slot])
        : "memory"
    );
    const uint32_t sequence = before + 1;
    #pragma unroll
    for (int dst = 0; dst < pk_fused_globals::NUM_DEVICES; ++dst) {
        asm volatile(
            "st.release.sys.global.u32 [%0], %1;"
            :: "l"(&barrier[dst][own_slot]), "r"(sequence)
            : "memory"
        );
    }
    const uint64_t wait_start = globaltimer();
    #pragma unroll
    for (int src = 0; src < pk_fused_globals::NUM_DEVICES; ++src) {
        uint32_t value;
        do {
            asm volatile(
                "ld.acquire.sys.global.u32 %0, [%1];"
                : "=r"(value)
                : "l"(&barrier[dev_idx][hierarchical_join_slot(src)])
                : "memory"
            );
            if (value >= sequence) break;
            if (globaltimer() - wait_start >= timeout_ns) {
                uint32_t ignored;
                const uint32_t code =
                    0x10000000u | static_cast<uint32_t>(src);
                asm volatile(
                    "atom.relaxed.sys.global.cas.b32 %0, [%1], %2, %3;"
                    : "=r"(ignored)
                    : "l"(error), "r"(0u), "r"(code)
                    : "memory"
                );
                return;
            }
        } while (true);
    }
}

template <int NUM_THREADS, bool INSTRUMENT, Fp4Mode MODE>
__global__ __launch_bounds__(NUM_THREADS, 1)
void fp4_communication_kernel(
    const __grid_constant__ fp4_globals G,
    WaitMode wait_mode,
    int64_t *trace,
    int *smids
) {
    constexpr int NUM_WARPS = NUM_THREADS / WARP_THREADS;
    static_assert(NUM_THREADS % WARP_THREADS == 0);

    if (threadIdx.x == 0) {
        trace_timestamp<INSTRUMENT>(trace, blockIdx.x, 0);
        smid_write<INSTRUMENT>(smids, blockIdx.x);
    }
    if (wait_mode == WaitMode::PDL_GRID) {
        // PDL only establishes same-device admission/completion.  The FP4
        // producer still has to join the eight-rank system-scope gate before
        // this grid reads peer data (PRE) or performs NVLS ld_reduce (POST).
        kittens::pdl::wait();
        if (threadIdx.x == 0) {
            wait_counter_acquire(
                G.barrier, grid_ready_count(), G.dev_idx,
                fp4_globals::NUM_DEVICES
            );
        }
        __syncthreads();
    }
    if (wait_mode == WaitMode::GRID_COUNTER) {
        if (threadIdx.x == 0) {
            wait_counter_acquire(
                G.barrier, grid_ready_count(), G.dev_idx,
                fp4_globals::NUM_DEVICES
            );
        }
        __syncthreads();
    }

    const int row_blocks = G.A.rows() / fp4_globals::ROW_BLOCK;
    const int col_blocks = G.B.cols() / fp4_globals::COL_BLOCK;
    const int super_rows = (row_blocks / fp4_globals::SUPER_M)
        * fp4_globals::SUPER_M;
    const int final_rows = row_blocks - super_rows;
    const int super_blocks = fp4_globals::SUPER_M * col_blocks;
    const int num_blocks = row_blocks * col_blocks;
    bool first_task = true;
    constexpr int TILE_READY_CONTRIBUTIONS = fp4_globals::NUM_DEVICES;

    if (threadIdx.x == 0) trace_timestamp<INSTRUMENT>(trace, blockIdx.x, 1);

    for (
        int task_id = fp4_globals::NUM_DEVICES * blockIdx.x + G.dev_idx;
        task_id < num_blocks;
        task_id += fp4_globals::NUM_DEVICES * gridDim.x
    ) {
        int row_idx, col_idx;
        task_to_tile(
            task_id, super_rows, final_rows, super_blocks, col_blocks,
            row_idx, col_idx
        );
        if (wait_mode == WaitMode::TILE_COUNTER) {
            if (threadIdx.x == 0) {
                wait_counter_acquire(
                    G.barrier, {row_idx, col_idx}, G.dev_idx,
                    TILE_READY_CONTRIBUTIONS
                );
            }
            __syncthreads();
        } else if (wait_mode == WaitMode::TILE_COUNTER_PER_RANK) {
            if (threadIdx.x == 0) {
                // External producers (for example a CuTe-DSL GEMM epilogue)
                // publish exactly once into their own local VMM allocation.
                // Acquire every peer separately instead of requiring those
                // producers to carry a peer-pointer table or multicast one
                // contribution into all eight target counters.
                #pragma unroll
                for (int rank = 0; rank < fp4_globals::NUM_DEVICES; ++rank) {
                    wait_counter_acquire(
                        G.barrier, {row_idx, col_idx}, rank, 1
                    );
                }
            }
            __syncthreads();
        }
        if (first_task && threadIdx.x == 0) {
            trace_timestamp<INSTRUMENT>(trace, blockIdx.x, 2);
        }

        constexpr int WARPS_PER_ROW = fp4_globals::COL_BLOCK / (WARP_THREADS * 2);
        constexpr int WORK_ITEMS = fp4_globals::ROW_BLOCK * WARPS_PER_ROW;
        const int lane = threadIdx.x % WARP_THREADS;
        const int warp = threadIdx.x / WARP_THREADS;
        for (int i = warp; i < WORK_ITEMS; i += NUM_WARPS) {
            const int global_row = row_idx * fp4_globals::ROW_BLOCK
                + i / WARPS_PER_ROW;
            const int global_col = col_idx * fp4_globals::COL_BLOCK
                + (i % WARPS_PER_ROW) * WARP_THREADS * 2
                + lane * 2;
            const int block_col = global_col / 16;
            const int byte_col = global_col / 2;

            if constexpr (MODE == Fp4Mode::PRE) {
                float sum0 = 0.0f;
                float sum1 = 0.0f;
                #pragma unroll
                for (int rank = 0; rank < fp4_globals::NUM_DEVICES; ++rank) {
                    const uint8_t packed = G.data[rank][{
                        0, rank, global_row, byte_col
                    }];
                    const float scale = fp4::scale_as_float(
                        G.scales[rank][{
                            0, rank, global_row, block_col
                        }]
                    );
                    sum0 += fp4::decode_nibble(packed & 0xf) * scale;
                    sum1 += fp4::decode_nibble(packed >> 4) * scale;
                }
                bf16_2 reduced = __float22bfloat162_rn(
                    make_float2(sum0, sum1)
                );
                auto *dst = reinterpret_cast<bf16_2 *>(
                    G.C.mc_ptr_at({0, 0, global_row, global_col})
                );
                multimem<bf16_2>::st(dst, reduced);
            } else {
                bf16_2 reduced;
                auto *dst = reinterpret_cast<bf16_2 *>(
                    G.C.mc_ptr_at({0, 0, global_row, global_col})
                );
                multimem<bf16_2>::ld_reduce<reduce_op::ADD>(reduced, dst);

                // POST is a communication-preserving epilogue: the BF16
                // all-reduce result remains available in C, while the FP4
                // representation is emitted as an additional diagnostic /
                // wire-format buffer.  The multicast store mirrors the
                // ordinary split communication path and makes C directly
                // comparable with the BF16 reference.
                multimem<bf16_2>::st(dst, reduced);

                const float first = __bfloat162float(reduced.x);
                const float second = __bfloat162float(reduced.y);
                const float local_max = fmaxf(fabsf(first), fabsf(second));
                const float amax = fp4::subgroup_max<8>(local_max);
                const bf16 scale = fp4::decode_scale(amax);
                const float scale_float = fp4::scale_as_float(scale);
                const uint8_t packed = fp4::encode_pair(
                    fp4::normalize(first, scale_float),
                    fp4::normalize(second, scale_float)
                );

                const uint32_t data_word = fp4::gather_u8x4(
                    packed, lane
                );
                if ((lane & 3) == 0) {
                    auto *data_dst = reinterpret_cast<uint32_t *>(
                        G.data.mc_ptr_at({
                            0, G.dev_idx, global_row, byte_col & ~3
                        })
                    );
                    fp4::multicast_store_u32(data_dst, data_word);
                }
                // multimem floating stores must be at least 32 bits on SM90.
                // Pair adjacent 16-element BF16 scales into one bf16x2-sized
                // bit-preserving multicast transaction.
                const uint32_t scale_bits =
                    static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(
                        &scale
                    ));
                // Every lane named by the full mask must execute the shuffle.
                // Lanes 0 and 16 consume the scale from the adjacent 8-lane
                // subgroup to form two consecutive 16-element scale records.
                const uint32_t adjacent_scale_bits = __shfl_xor_sync(
                    0xffffffffu, scale_bits, 8, 16
                );
                if ((lane & 15) == 0) {
                    const uint32_t packed_scales =
                        scale_bits | (adjacent_scale_bits << 16);
                    auto *scale_dst = reinterpret_cast<uint32_t *>(
                        G.scales.mc_ptr_at({
                            0, G.dev_idx, global_row, block_col
                        })
                    );
                    fp4::multicast_store_u32(scale_dst, packed_scales);
                }
            }
        }
        first_task = false;
    }
    if (threadIdx.x == 0) trace_timestamp<INSTRUMENT>(trace, blockIdx.x, 3);
}

struct StreamState {
    int device = -1;
    cudaStream_t communication = nullptr;
    cudaEvent_t launch_ready = nullptr;
    cudaEvent_t communication_done = nullptr;
};

StreamState &stream_state() {
    static StreamState state;
    static std::mutex lock;
    std::lock_guard<std::mutex> guard(lock);

    int device;
    CUDACHECK(cudaGetDevice(&device));
    if (state.device == device && state.communication != nullptr) return state;

    if (state.communication != nullptr) {
        CUDACHECK(cudaEventDestroy(state.launch_ready));
        CUDACHECK(cudaEventDestroy(state.communication_done));
        CUDACHECK(cudaStreamDestroy(state.communication));
    }
    state.device = device;
    CUDACHECK(cudaStreamCreateWithFlags(
        &state.communication, cudaStreamNonBlocking
    ));
    CUDACHECK(cudaEventCreateWithFlags(
        &state.launch_ready, cudaEventDisableTiming
    ));
    CUDACHECK(cudaEventCreateWithFlags(
        &state.communication_done, cudaEventDisableTiming
    ));
    return state;
}

template <int NUM_THREADS, bool INSTRUMENT>
void launch_communication_typed(
    const pk_fused_globals &G,
    int num_ctas,
    int dynamic_smem,
    WaitMode wait_mode,
    bool enable_pdl,
    cudaStream_t stream,
    uint32_t *resident_counter,
    int64_t *trace,
    int *smids
) {
    auto kernel = split_communication_kernel<NUM_THREADS, INSTRUMENT>;
    CUDACHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_smem
    ));
    if (enable_pdl) {
        kittens::LaunchConfig<false, true> launch_config(
            dim3(num_ctas), dim3(NUM_THREADS), dynamic_smem, stream
        );
        CUDACHECK(cudaLaunchKernelEx(
            launch_config, kernel, G, wait_mode, resident_counter, trace, smids
        ));
    } else {
        kernel<<<num_ctas, NUM_THREADS, dynamic_smem, stream>>>(
            G, wait_mode, resident_counter, trace, smids
        );
        CUDACHECK(cudaGetLastError());
    }
}

template <int NUM_THREADS>
void launch_communication_selected(
    const pk_fused_globals &G,
    int num_ctas,
    int dynamic_smem,
    WaitMode wait_mode,
    bool enable_pdl,
    bool instrument,
    cudaStream_t stream,
    uint32_t *resident_counter,
    int64_t *trace,
    int *smids
) {
    if (instrument) {
        launch_communication_typed<NUM_THREADS, true>(
            G, num_ctas, dynamic_smem, wait_mode, enable_pdl, stream,
            resident_counter, trace, smids
        );
    } else {
        launch_communication_typed<NUM_THREADS, false>(
            G, num_ctas, dynamic_smem, wait_mode, enable_pdl, stream,
            resident_counter, nullptr, nullptr
        );
    }
}

void launch_communication(
    const pk_fused_globals &G,
    int num_ctas,
    int num_threads,
    int dynamic_smem,
    WaitMode wait_mode,
    bool enable_pdl,
    bool instrument,
    cudaStream_t stream,
    uint32_t *resident_counter,
    int64_t *trace,
    int *smids
) {
    switch (num_threads) {
        case 256:
            launch_communication_selected<256>(
                G, num_ctas, dynamic_smem, wait_mode, enable_pdl,
                instrument, stream, resident_counter, trace, smids
            );
            break;
        case 384:
            launch_communication_selected<384>(
                G, num_ctas, dynamic_smem, wait_mode, enable_pdl,
                instrument, stream, resident_counter, trace, smids
            );
            break;
        case 512:
            launch_communication_selected<512>(
                G, num_ctas, dynamic_smem, wait_mode, enable_pdl,
                instrument, stream, resident_counter, trace, smids
            );
            break;
        case 768:
            launch_communication_selected<768>(
                G, num_ctas, dynamic_smem, wait_mode, enable_pdl,
                instrument, stream, resident_counter, trace, smids
            );
            break;
        case 1024:
            launch_communication_selected<1024>(
                G, num_ctas, dynamic_smem, wait_mode, enable_pdl,
                instrument, stream, resident_counter, trace, smids
            );
            break;
        default:
            TORCH_CHECK(
                false,
                "comm_threads must be one of 256, 384, 512, 768, 1024"
            );
    }
}

template <int NUM_THREADS>
void launch_hierarchical_pack_typed(
    const pk_fused_globals &G,
    bf16 *wire,
    uint32_t *ready,
    uint32_t *error,
    uint32_t epoch,
    int num_ctas,
    WaitMode wait_mode,
    bool enable_pdl,
    cudaStream_t stream
) {
    auto kernel = hierarchical_pack_kernel<NUM_THREADS>;
    if (enable_pdl) {
        kittens::LaunchConfig<false, true> config(
            dim3(num_ctas), dim3(NUM_THREADS), 0, stream
        );
        CUDACHECK(cudaLaunchKernelEx(
            config,
            kernel,
            G,
            wire,
            ready,
            error,
            epoch,
            wait_mode
        ));
    } else {
        kernel<<<num_ctas, NUM_THREADS, 0, stream>>>(
            G, wire, ready, error, epoch, wait_mode
        );
        CUDACHECK(cudaGetLastError());
    }
}

void launch_hierarchical_pack(
    const pk_fused_globals &G,
    bf16 *wire,
    uint32_t *ready,
    uint32_t *error,
    uint32_t epoch,
    int num_ctas,
    int num_threads,
    WaitMode wait_mode,
    bool enable_pdl,
    cudaStream_t stream
) {
    switch (num_threads) {
        case 256:
            launch_hierarchical_pack_typed<256>(
                G, wire, ready, error, epoch, num_ctas,
                wait_mode, enable_pdl, stream
            );
            break;
        case 512:
            launch_hierarchical_pack_typed<512>(
                G, wire, ready, error, epoch, num_ctas,
                wait_mode, enable_pdl, stream
            );
            break;
        case 1024:
            launch_hierarchical_pack_typed<1024>(
                G, wire, ready, error, epoch, num_ctas,
                wait_mode, enable_pdl, stream
            );
            break;
        default:
            TORCH_CHECK(false, "pack_threads must be 256, 512, or 1024");
    }
}

template <int NUM_THREADS>
void launch_hierarchical_unpack_typed(
    const pk_fused_globals::C_pgl &C,
    const bf16 *wire,
    int dev_idx,
    int row_blocks,
    int col_blocks,
    int first_slot,
    int slot_count,
    cudaStream_t stream
) {
    hierarchical_unpack_kernel<NUM_THREADS><<<
        slot_count, NUM_THREADS, 0, stream
    >>>(
        C,
        wire,
        dev_idx,
        row_blocks,
        col_blocks,
        first_slot,
        slot_count
    );
    CUDACHECK(cudaGetLastError());
}

void launch_hierarchical_unpack(
    const pk_fused_globals::C_pgl &C,
    const bf16 *wire,
    int dev_idx,
    int row_blocks,
    int col_blocks,
    int first_slot,
    int slot_count,
    int num_threads,
    cudaStream_t stream
) {
    switch (num_threads) {
        case 256:
            launch_hierarchical_unpack_typed<256>(
                C, wire, dev_idx, row_blocks, col_blocks,
                first_slot, slot_count, stream
            );
            break;
        case 512:
            launch_hierarchical_unpack_typed<512>(
                C, wire, dev_idx, row_blocks, col_blocks,
                first_slot, slot_count, stream
            );
            break;
        case 1024:
            launch_hierarchical_unpack_typed<1024>(
                C, wire, dev_idx, row_blocks, col_blocks,
                first_slot, slot_count, stream
            );
            break;
        default:
            TORCH_CHECK(false, "unpack_threads must be 256, 512, or 1024");
    }
}

void launch_compute(
    const pk_fused_globals &G,
    int num_ctas,
    bool trigger_pdl,
    bool publish_grid_ready,
    bool instrument,
    cudaStream_t stream,
    int64_t *trace,
    int *smids
) {
    if (instrument) {
        auto kernel = split_compute_kernel<true>;
        CUDACHECK(cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            pk_fused_config::DYNAMIC_SHARED_MEMORY
        ));
        kernel<<<
            num_ctas,
            pk_fused_config::NUM_THREADS,
            pk_fused_config::DYNAMIC_SHARED_MEMORY,
            stream
        >>>(
            G, trigger_pdl, publish_grid_ready, trace, smids
        );
    } else {
        auto kernel = split_compute_kernel<false>;
        CUDACHECK(cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            pk_fused_config::DYNAMIC_SHARED_MEMORY
        ));
        kernel<<<
            num_ctas,
            pk_fused_config::NUM_THREADS,
            pk_fused_config::DYNAMIC_SHARED_MEMORY,
            stream
        >>>(
            G, trigger_pdl, publish_grid_ready, nullptr, nullptr
        );
    }
    CUDACHECK(cudaGetLastError());
}

void launch_fp4_pre_compute(
    const fp4_globals &G,
    int num_ctas,
    bool trigger_pdl,
    bool publish_grid_ready,
    bool instrument,
    cudaStream_t stream,
    int64_t *trace,
    int *smids
) {
    if (instrument) {
        auto kernel = fp4_pre_compute_kernel<true>;
        CUDACHECK(cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            FP4_DYNAMIC_SHARED_MEMORY
        ));
        if (trigger_pdl) {
            kittens::LaunchConfig<false, true> config(
                dim3(num_ctas), dim3(pk_fused_config::NUM_THREADS),
                FP4_DYNAMIC_SHARED_MEMORY, stream
            );
            CUDACHECK(cudaLaunchKernelEx(
                config, kernel, G, trigger_pdl, publish_grid_ready, trace, smids
            ));
        } else {
            kernel<<<
                num_ctas, pk_fused_config::NUM_THREADS,
                FP4_DYNAMIC_SHARED_MEMORY, stream
            >>>(G, trigger_pdl, publish_grid_ready, trace, smids);
        }
    } else {
        auto kernel = fp4_pre_compute_kernel<false>;
        CUDACHECK(cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            FP4_DYNAMIC_SHARED_MEMORY
        ));
        if (trigger_pdl) {
            kittens::LaunchConfig<false, true> config(
                dim3(num_ctas), dim3(pk_fused_config::NUM_THREADS),
                FP4_DYNAMIC_SHARED_MEMORY, stream
            );
            CUDACHECK(cudaLaunchKernelEx(
                config, kernel, G, trigger_pdl, publish_grid_ready,
                nullptr, nullptr
            ));
        } else {
            kernel<<<
                num_ctas, pk_fused_config::NUM_THREADS,
                FP4_DYNAMIC_SHARED_MEMORY, stream
            >>>(G, trigger_pdl, publish_grid_ready, nullptr, nullptr);
        }
    }
    CUDACHECK(cudaGetLastError());
}

template <int NUM_THREADS, bool INSTRUMENT, Fp4Mode MODE>
void launch_fp4_communication_typed(
    const fp4_globals &G,
    int num_ctas,
    WaitMode wait_mode,
    bool enable_pdl,
    cudaStream_t stream,
    int64_t *trace,
    int *smids
) {
    auto kernel = fp4_communication_kernel<NUM_THREADS, INSTRUMENT, MODE>;
    if (enable_pdl) {
        kittens::LaunchConfig<false, true> config(
            dim3(num_ctas), dim3(NUM_THREADS), 0, stream
        );
        CUDACHECK(cudaLaunchKernelEx(
            config, kernel, G, wait_mode, trace, smids
        ));
    } else {
        kernel<<<num_ctas, NUM_THREADS, 0, stream>>>(
            G, wait_mode, trace, smids
        );
    }
}

template <int NUM_THREADS, Fp4Mode MODE>
void launch_fp4_communication_selected(
    const fp4_globals &G,
    int num_ctas,
    WaitMode wait_mode,
    bool enable_pdl,
    bool instrument,
    cudaStream_t stream,
    int64_t *trace,
    int *smids
) {
    if (instrument) {
        launch_fp4_communication_typed<NUM_THREADS, true, MODE>(
            G, num_ctas, wait_mode, enable_pdl, stream, trace, smids
        );
    } else {
        launch_fp4_communication_typed<NUM_THREADS, false, MODE>(
            G, num_ctas, wait_mode, enable_pdl, stream, nullptr, nullptr
        );
    }
}

template <Fp4Mode MODE>
void launch_fp4_communication(
    const fp4_globals &G,
    int num_ctas,
    int num_threads,
    WaitMode wait_mode,
    bool enable_pdl,
    bool instrument,
    cudaStream_t stream,
    int64_t *trace,
    int *smids
) {
    switch (num_threads) {
        case 256:
            launch_fp4_communication_selected<256, MODE>(
                G, num_ctas, wait_mode, enable_pdl, instrument,
                stream, trace, smids
            );
            break;
        case 384:
            launch_fp4_communication_selected<384, MODE>(
                G, num_ctas, wait_mode, enable_pdl, instrument,
                stream, trace, smids
            );
            break;
        case 512:
            launch_fp4_communication_selected<512, MODE>(
                G, num_ctas, wait_mode, enable_pdl, instrument,
                stream, trace, smids
            );
            break;
        case 768:
            launch_fp4_communication_selected<768, MODE>(
                G, num_ctas, wait_mode, enable_pdl, instrument,
                stream, trace, smids
            );
            break;
        case 1024:
            launch_fp4_communication_selected<1024, MODE>(
                G, num_ctas, wait_mode, enable_pdl, instrument,
                stream, trace, smids
            );
            break;
        default:
            TORCH_CHECK(false, "unsupported FP4 communication thread count");
    }
    CUDACHECK(cudaGetLastError());
}

void launch_reset(const pk_fused_globals &G, cudaStream_t stream) {
    CUDACHECK(cudaFuncSetAttribute(
        kittens::py::global_kernel<
            pk_fused_config,
            pk_fused_globals,
            pk_fused_epilogue_kernel
        >,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        pk_fused_config::DYNAMIC_SHARED_MEMORY
    ));
    kittens::py::global_kernel<
        pk_fused_config,
        pk_fused_globals,
        pk_fused_epilogue_kernel
    ><<<
        pk_fused_config::NUM_BLOCKS,
        pk_fused_config::NUM_THREADS,
        pk_fused_config::DYNAMIC_SHARED_MEMORY,
        stream
    >>>(G);
    CUDACHECK(cudaGetLastError());
}

__device__ inline void split_epilogue_kernel(const pk_fused_globals &G) {
    const int row_blocks = G.A.rows() / pk_fused_globals::ROW_BLOCK;
    const int col_blocks = G.B.cols() / pk_fused_globals::COL_BLOCK;
    const int num_blocks = row_blocks * col_blocks;
    const int offset = threadIdx.x;
    const int stride = blockDim.x;
    if (threadIdx.x == 0) {
        barrier_all(G.barrier, reset_pre_count(), G.dev_idx);
    }
    __syncthreads();
    for (int i = offset; i < num_blocks; i += stride) {
        G.barrier[G.dev_idx][{i / col_blocks, i % col_blocks}] = 0;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        G.barrier[G.dev_idx][grid_local_count()] = 0;
        G.barrier[G.dev_idx][grid_ready_count()] = 0;
        barrier_all(G.barrier, reset_post_count(), G.dev_idx);
    }
}

__global__ void split_reset_kernel(
    const __grid_constant__ pk_fused_globals G
) {
    split_epilogue_kernel(G);
}

void launch_split_reset(const pk_fused_globals &G, cudaStream_t stream) {
    split_reset_kernel<<<1, 256, 0, stream>>>(G);
    CUDACHECK(cudaGetLastError());
}

__device__ inline void fp4_reset_body(const fp4_globals &G) {
    const int row_blocks = G.A.rows() / fp4_globals::ROW_BLOCK;
    const int col_blocks = G.B.cols() / fp4_globals::COL_BLOCK;
    const int num_blocks = row_blocks * col_blocks;
    const int offset = threadIdx.x;
    const int stride = blockDim.x;
    if (threadIdx.x == 0) {
        barrier_all(G.barrier, reset_pre_count(), G.dev_idx);
    }
    __syncthreads();
    for (int i = offset; i < num_blocks; i += stride) {
        G.barrier[G.dev_idx][{i / col_blocks, i % col_blocks}] = 0;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        G.barrier[G.dev_idx][grid_local_count()] = 0;
        G.barrier[G.dev_idx][grid_ready_count()] = 0;
        barrier_all(G.barrier, reset_post_count(), G.dev_idx);
    }
}

__global__
void fp4_reset_kernel(const __grid_constant__ fp4_globals G) {
    fp4_reset_body(G);
}

void launch_fp4_reset(const fp4_globals &G, cudaStream_t stream) {
    fp4_reset_kernel<<<1, 256, 0, stream>>>(G);
    CUDACHECK(cudaGetLastError());
}

LaunchMode parse_mode(const std::string &mode) {
    if (mode == "default") return LaunchMode::DEFAULT_STREAM;
    if (mode == "pdl_grid") return LaunchMode::PDL_GRID;
    if (mode == "pdl_tile") return LaunchMode::PDL_TILE;
    if (mode == "two_stream") return LaunchMode::TWO_STREAM;
    TORCH_CHECK(
        false,
        "mode must be default, pdl_grid, pdl_tile, or two_stream"
    );
}

void check_trace_tensor(
    const at::Tensor &tensor,
    at::ScalarType dtype,
    int min_elements,
    const char *name
) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has wrong dtype");
    TORCH_CHECK(tensor.numel() >= min_elements, name, " is too small");
}

void check_hierarchical_epoch(uint32_t epoch) {
    TORCH_CHECK(
        epoch > 0 && epoch <= MAX_HIERARCHICAL_EPOCH,
        "hierarchical epoch must be in 1..2^31-1"
    );
}

void check_hierarchical_threads(int threads, const char *name) {
    TORCH_CHECK(
        threads == 256 || threads == 512 || threads == 1024,
        name, " must be 256, 512, or 1024"
    );
}

int hierarchical_owner_slots(int num_blocks, int local_rank) {
    TORCH_CHECK(
        local_rank >= 0 && local_rank < pk_fused_globals::NUM_DEVICES,
        "hierarchical local rank must be in the eight-GPU NVLS group"
    );
    TORCH_CHECK(num_blocks > local_rank, "local rank owns no output tile");
    return (
        num_blocks + pk_fused_globals::NUM_DEVICES - 1 - local_rank
    ) / pk_fused_globals::NUM_DEVICES;
}

int check_hierarchical_buffers(
    const pk_fused_globals &G,
    const at::Tensor &wire,
    const at::Tensor &ready,
    const at::Tensor &error
) {
    const int num_blocks = G.C.rows() / pk_fused_globals::ROW_BLOCK
        * (G.C.cols() / pk_fused_globals::COL_BLOCK);
    const int num_slots = hierarchical_owner_slots(num_blocks, G.dev_idx);
    TORCH_CHECK(
        wire.numel()
            >= static_cast<int64_t>(num_slots) * HIERARCHICAL_TILE_ELEMENTS,
        "wire is too small for the owner tiles"
    );
    TORCH_CHECK(ready.numel() >= num_slots, "ready has too few owner slots");
    TORCH_CHECK(error.numel() >= 1, "error must contain at least one word");
    return num_slots;
}

pk_fused_globals make_globals(
    const at::Tensor &A,
    const at::Tensor &B,
    kittens::py::TKParallelTensor &C,
    kittens::py::TKParallelTensor &barrier,
    int num_comp_ctas,
    int num_comm_ctas
) {
    kittens::py::device_check(A, B, C.data_, barrier.data_);
    kittens::py::parallel_tensor_check(C, barrier);
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(
        A.scalar_type() == at::ScalarType::BFloat16
            && B.scalar_type() == at::ScalarType::BFloat16,
        "A and B must be BF16"
    );
    TORCH_CHECK(A.size(1) == B.size(0), "incompatible GEMM dimensions");
    TORCH_CHECK(
        C.data_.dim() == 2 && C.data_.size(0) == A.size(0)
            && C.data_.size(1) == B.size(1)
            && C.data_.scalar_type() == at::ScalarType::BFloat16,
        "C must be BF16 with shape [M, N]"
    );
    TORCH_CHECK(
        barrier.data_.dim() == 3 && barrier.data_.size(0) >= 2
            && barrier.data_.scalar_type() == at::ScalarType::Int,
        "barrier must be INT32 with shape [>=2, rows, cols]"
    );
    TORCH_CHECK(A.size(0) % pk_fused_globals::ROW_BLOCK == 0);
    TORCH_CHECK(B.size(1) % pk_fused_globals::COL_BLOCK == 0);
    TORCH_CHECK(A.size(1) % pk_fused_globals::RED_BLOCK == 0);
    const int64_t row_blocks = A.size(0) / pk_fused_globals::ROW_BLOCK;
    const int64_t col_blocks = B.size(1) / pk_fused_globals::COL_BLOCK;
    TORCH_CHECK(
        barrier.data_.size(1) >= row_blocks
            && barrier.data_.size(2) >= std::max<int64_t>(col_blocks, 13),
        "barrier is too small for tile and grid protocol counters"
    );
    TORCH_CHECK(
        A.size(0) <= std::numeric_limits<int>::max()
            && A.size(1) <= std::numeric_limits<int>::max()
            && B.size(1) <= std::numeric_limits<int>::max()
            && row_blocks * col_blocks <= std::numeric_limits<int>::max(),
        "GEMM shape exceeds the 32-bit kernel indexing contract"
    );
    TORCH_CHECK(
        A.size(1) / pk_fused_globals::RED_BLOCK
            >= pk_fused_globals::PIPELINE_STAGES,
        "K must contain at least four reduction tiles"
    );
    TORCH_CHECK(num_comp_ctas > 0 && num_comp_ctas <= NUM_SMS);
    TORCH_CHECK(num_comm_ctas > 0 && num_comm_ctas <= NUM_SMS);

    return pk_fused_globals {
        .A = kittens::py::tensor_to_gl<pk_fused_globals::A_gl>(A),
        .B = kittens::py::tensor_to_gl<pk_fused_globals::B_gl>(B),
        .C = kittens::py::parallel_tensor_to_pgl<pk_fused_globals::C_pgl>(C),
        .barrier = kittens::py::parallel_tensor_to_pgl<
            pk_fused_globals::barrier_pgl
        >(barrier),
        .dev_idx = barrier.local_rank_,
        .num_comm_sms = num_comm_ctas,
        .num_comp_sms = num_comp_ctas,
    };
}

fp4_globals make_fp4_globals(
    const at::Tensor &A,
    const at::Tensor &B,
    kittens::py::TKParallelTensor &C,
    kittens::py::TKParallelTensor &data,
    kittens::py::TKParallelTensor &scales,
    kittens::py::TKParallelTensor &barrier,
    int num_comp_ctas,
    int num_comm_ctas
) {
    kittens::py::device_check(
        A, B, C.data_, data.data_, scales.data_, barrier.data_
    );
    kittens::py::parallel_tensor_check(C, data, scales, barrier);
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "incompatible GEMM dimensions");
    TORCH_CHECK(
        C.data_.dim() == 2 && C.data_.size(0) == A.size(0)
            && C.data_.size(1) == B.size(1),
        "BF16 output must have shape [M, N]"
    );
    TORCH_CHECK(
        data.data_.dim() == 3
            && data.data_.size(0) == fp4_globals::NUM_DEVICES
            && data.data_.size(1) == A.size(0)
            && data.data_.size(2) == B.size(1) / 2,
        "FP4 data must have shape [8, M, N/2]"
    );
    TORCH_CHECK(
        scales.data_.dim() == 3
            && scales.data_.size(0) == fp4_globals::NUM_DEVICES
            && scales.data_.size(1) == A.size(0)
            && scales.data_.size(2) == B.size(1) / 16,
        "FP4 scales must have shape [8, M, N/16]"
    );
    TORCH_CHECK(A.size(0) % fp4_globals::ROW_BLOCK == 0);
    TORCH_CHECK(B.size(1) % fp4_globals::COL_BLOCK == 0);
    TORCH_CHECK(A.size(1) % fp4_globals::RED_BLOCK == 0);
    TORCH_CHECK(
        A.size(1) / fp4_globals::RED_BLOCK >= fp4_globals::PIPELINE_STAGES,
        "K must contain at least four reduction tiles"
    );
    TORCH_CHECK(num_comp_ctas > 0 && num_comp_ctas <= NUM_SMS);
    TORCH_CHECK(num_comm_ctas > 0 && num_comm_ctas <= NUM_SMS);

    return fp4_globals{
        .A = kittens::py::tensor_to_gl<fp4_globals::A_gl>(A),
        .B = kittens::py::tensor_to_gl<fp4_globals::B_gl>(B),
        .C = kittens::py::parallel_tensor_to_pgl<fp4_globals::C_pgl>(C),
        .data = kittens::py::parallel_tensor_to_pgl<fp4_globals::data_pgl>(
            data
        ),
        .scales = kittens::py::parallel_tensor_to_pgl<fp4_globals::scale_pgl>(
            scales
        ),
        .barrier = kittens::py::parallel_tensor_to_pgl<
            fp4_globals::barrier_pgl
        >(barrier),
        .dev_idx = barrier.local_rank_,
        .num_comm_sms = num_comm_ctas,
        .num_comp_sms = num_comp_ctas,
    };
}

void split_entrypoint(
    const at::Tensor &A,
    const at::Tensor &B,
    kittens::py::TKParallelTensor &C,
    kittens::py::TKParallelTensor &barrier,
    const std::string &mode_string,
    int num_comp_ctas,
    int num_comm_ctas,
    int comm_threads,
    int comm_smem_bytes,
    bool instrument,
    const at::Tensor &compute_trace,
    const at::Tensor &communication_trace,
    const at::Tensor &compute_smids,
    const at::Tensor &communication_smids
) {
    c10::cuda::CUDAGuard guard(A.device());
    auto mode = parse_mode(mode_string);
    TORCH_CHECK(comm_smem_bytes >= 0);
    TORCH_CHECK(comm_smem_bytes <= MAX_SHARED_MEMORY);
    if (mode == LaunchMode::TWO_STREAM) {
        TORCH_CHECK(
            num_comm_ctas < NUM_SMS,
            "two_stream requires at least one SM left for compute"
        );
        TORCH_CHECK(
            num_comp_ctas + num_comm_ctas <= NUM_SMS,
            "two_stream compute and communication CTA reservations exceed "
            "the H100 SM count"
        );
    }

    if (instrument) {
        check_trace_tensor(
            compute_trace, at::ScalarType::Long,
            num_comp_ctas * TRACE_POINTS, "compute_trace"
        );
        check_trace_tensor(
            communication_trace, at::ScalarType::Long,
            num_comm_ctas * TRACE_POINTS, "communication_trace"
        );
        check_trace_tensor(
            compute_smids, at::ScalarType::Int,
            num_comp_ctas, "compute_smids"
        );
        check_trace_tensor(
            communication_smids, at::ScalarType::Int,
            num_comm_ctas, "communication_smids"
        );
    }

    auto G = make_globals(
        A, B, C, barrier, num_comp_ctas, num_comm_ctas
    );
    auto main_stream = at::cuda::getCurrentCUDAStream().stream();
    auto *compute_trace_ptr = instrument
        ? compute_trace.data_ptr<int64_t>() : nullptr;
    auto *communication_trace_ptr = instrument
        ? communication_trace.data_ptr<int64_t>() : nullptr;
    auto *compute_smids_ptr = instrument
        ? compute_smids.data_ptr<int>() : nullptr;
    auto *communication_smids_ptr = instrument
        ? communication_smids.data_ptr<int>() : nullptr;

    if (mode == LaunchMode::TWO_STREAM) {
        auto &state = stream_state();

        // Anchor the non-blocking communication stream to the current stream.
        // The Python timing start event is already queued on main_stream when
        // this entrypoint is called.  Without this rendezvous, CUDA may admit
        // the communication grid before that external event executes, which
        // under-counts the two-stream critical path even though the host
        // submitted the compute launch first.
        CUDACHECK(cudaEventRecord(state.launch_ready, main_stream));
        CUDACHECK(cudaStreamWaitEvent(
            state.communication, state.launch_ready, 0
        ));
        launch_compute(
            G, num_comp_ctas, false, false, instrument, main_stream,
            compute_trace_ptr, compute_smids_ptr
        );
        launch_communication(
            G, num_comm_ctas, comm_threads, comm_smem_bytes,
            WaitMode::TILE_COUNTER, false, instrument,
            state.communication,
            nullptr,
            communication_trace_ptr, communication_smids_ptr
        );
        CUDACHECK(cudaEventRecord(
            state.communication_done, state.communication
        ));
        CUDACHECK(cudaStreamWaitEvent(
            main_stream, state.communication_done, 0
        ));
    } else {
        launch_compute(
            G, num_comp_ctas, mode != LaunchMode::DEFAULT_STREAM,
            mode == LaunchMode::DEFAULT_STREAM
                || mode == LaunchMode::PDL_GRID,
            instrument, main_stream,
            compute_trace_ptr, compute_smids_ptr
        );
        WaitMode wait_mode = WaitMode::GRID_COUNTER;
        bool enable_pdl = false;
        if (mode == LaunchMode::PDL_GRID) {
            wait_mode = WaitMode::PDL_GRID;
            enable_pdl = true;
        } else if (mode == LaunchMode::PDL_TILE) {
            wait_mode = WaitMode::TILE_COUNTER;
            enable_pdl = true;
        }
        launch_communication(
            G, num_comm_ctas, comm_threads, comm_smem_bytes,
            wait_mode, enable_pdl, instrument, main_stream, nullptr,
            communication_trace_ptr, communication_smids_ptr
        );
    }

    launch_split_reset(G, main_stream);
}

void local_bf16_entrypoint(
    const at::Tensor &A,
    const at::Tensor &B,
    kittens::py::TKParallelTensor &C,
    kittens::py::TKParallelTensor &barrier,
    int num_comp_ctas
) {
    c10::cuda::CUDAGuard guard(A.device());
    auto G = make_globals(A, B, C, barrier, num_comp_ctas, 1);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_compute(
        G, num_comp_ctas, false, false, false, stream, nullptr, nullptr
    );
    launch_split_reset(G, stream);
}

void hierarchical_bf16_producer_entrypoint(
    const at::Tensor &A,
    const at::Tensor &B,
    kittens::py::TKParallelTensor &C,
    kittens::py::TKParallelTensor &barrier,
    const at::Tensor &wire,
    const at::Tensor &ready,
    const at::Tensor &error,
    const std::string &mode_string,
    int num_comp_ctas,
    int num_pack_ctas,
    int pack_threads,
    uint32_t epoch
) {
    c10::cuda::CUDAGuard guard(A.device());
    auto mode = parse_mode(mode_string);
    TORCH_CHECK(mode != LaunchMode::TWO_STREAM);
    check_hierarchical_epoch(epoch);
    TORCH_CHECK(num_pack_ctas > 0 && num_pack_ctas <= NUM_SMS);
    check_hierarchical_threads(pack_threads, "pack_threads");
    check_trace_tensor(wire, at::ScalarType::BFloat16, 1, "wire");
    check_trace_tensor(ready, at::ScalarType::Int, 1, "ready");
    check_trace_tensor(error, at::ScalarType::Int, 1, "error");
    kittens::py::device_check(
        A, B, C.data_, barrier.data_, wire, ready, error
    );

    auto G = make_globals(A, B, C, barrier, num_comp_ctas, num_pack_ctas);
    check_hierarchical_buffers(G, wire, ready, error);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_compute(
        G,
        num_comp_ctas,
        mode != LaunchMode::DEFAULT_STREAM,
        mode == LaunchMode::DEFAULT_STREAM || mode == LaunchMode::PDL_GRID,
        false,
        stream,
        nullptr,
        nullptr
    );
    WaitMode wait_mode = WaitMode::GRID_COUNTER;
    bool enable_pdl = false;
    if (mode == LaunchMode::PDL_GRID) {
        wait_mode = WaitMode::PDL_GRID;
        enable_pdl = true;
    } else if (mode == LaunchMode::PDL_TILE) {
        wait_mode = WaitMode::TILE_COUNTER;
        enable_pdl = true;
    }
    launch_hierarchical_pack(
        G,
        reinterpret_cast<bf16 *>(wire.data_ptr()),
        reinterpret_cast<uint32_t *>(ready.data_ptr<int>()),
        reinterpret_cast<uint32_t *>(error.data_ptr<int>()),
        epoch,
        num_pack_ctas,
        pack_threads,
        wait_mode,
        enable_pdl,
        stream
    );
    launch_split_reset(G, stream);
}

void hierarchical_bf16_unpack_entrypoint(
    kittens::py::TKParallelTensor &C,
    const at::Tensor &wire,
    int first_slot,
    int slot_count,
    int unpack_threads
) {
    c10::cuda::CUDAGuard guard(C.data_.device());
    check_trace_tensor(wire, at::ScalarType::BFloat16, 1, "wire");
    kittens::py::device_check(C.data_, wire);
    kittens::py::parallel_tensor_check<pk_fused_globals::C_pgl>(C);
    TORCH_CHECK(first_slot >= 0 && slot_count > 0);
    check_hierarchical_threads(unpack_threads, "unpack_threads");
    TORCH_CHECK(
        C.data_.dim() == 2
            && C.data_.size(0) % pk_fused_globals::ROW_BLOCK == 0
            && C.data_.size(1) % pk_fused_globals::COL_BLOCK == 0,
        "C shape must align to the ParallelKittens output tile"
    );
    const int64_t row_blocks64 =
        C.data_.size(0) / pk_fused_globals::ROW_BLOCK;
    const int64_t col_blocks64 =
        C.data_.size(1) / pk_fused_globals::COL_BLOCK;
    TORCH_CHECK(
        row_blocks64 * col_blocks64 <= std::numeric_limits<int>::max(),
        "C tile count exceeds the 32-bit unpack indexing contract"
    );
    const int row_blocks = static_cast<int>(row_blocks64);
    const int col_blocks = static_cast<int>(col_blocks64);
    const int num_blocks = row_blocks * col_blocks;
    const int num_slots = hierarchical_owner_slots(num_blocks, C.local_rank_);
    TORCH_CHECK(
        first_slot <= num_slots - slot_count,
        "unpack window exceeds the owner wire"
    );
    TORCH_CHECK(
        wire.numel()
            >= static_cast<int64_t>(num_slots) * HIERARCHICAL_TILE_ELEMENTS
    );
    auto C_pgl = kittens::py::parallel_tensor_to_pgl<
        pk_fused_globals::C_pgl
    >(C);
    launch_hierarchical_unpack(
        C_pgl,
        reinterpret_cast<const bf16 *>(wire.data_ptr()),
        C.local_rank_,
        row_blocks,
        col_blocks,
        first_slot,
        slot_count,
        unpack_threads,
        at::cuda::getCurrentCUDAStream().stream()
    );
}

void hierarchical_bf16_pack_only_entrypoint(
    const at::Tensor &A,
    const at::Tensor &B,
    kittens::py::TKParallelTensor &C,
    kittens::py::TKParallelTensor &barrier,
    const at::Tensor &wire,
    const at::Tensor &ready,
    const at::Tensor &error,
    int num_pack_ctas,
    int pack_threads,
    uint32_t epoch
) {
    c10::cuda::CUDAGuard guard(A.device());
    check_hierarchical_epoch(epoch);
    TORCH_CHECK(num_pack_ctas > 0 && num_pack_ctas <= NUM_SMS);
    check_hierarchical_threads(pack_threads, "pack_threads");
    check_trace_tensor(wire, at::ScalarType::BFloat16, 1, "wire");
    check_trace_tensor(ready, at::ScalarType::Int, 1, "ready");
    check_trace_tensor(error, at::ScalarType::Int, 1, "error");
    kittens::py::device_check(
        A, B, C.data_, barrier.data_, wire, ready, error
    );
    auto G = make_globals(A, B, C, barrier, 1, num_pack_ctas);
    check_hierarchical_buffers(G, wire, ready, error);
    launch_hierarchical_pack(
        G,
        reinterpret_cast<bf16 *>(wire.data_ptr()),
        reinterpret_cast<uint32_t *>(ready.data_ptr<int>()),
        reinterpret_cast<uint32_t *>(error.data_ptr<int>()),
        epoch,
        num_pack_ctas,
        pack_threads,
        WaitMode::NONE,
        false,
        at::cuda::getCurrentCUDAStream().stream()
    );
}

void hierarchical_wait_ready_entrypoint(
    const at::Tensor &ready,
    const at::Tensor &error,
    int first_slot,
    int slot_count,
    uint32_t epoch,
    uint64_t timeout_ns
) {
    c10::cuda::CUDAGuard guard(ready.device());
    check_trace_tensor(ready, at::ScalarType::Int, 1, "ready");
    check_trace_tensor(error, at::ScalarType::Int, 1, "error");
    kittens::py::device_check(ready, error);
    TORCH_CHECK(first_slot >= 0 && slot_count > 0 && slot_count <= 1024);
    TORCH_CHECK(
        static_cast<int64_t>(first_slot) + slot_count <= ready.numel(),
        "ready window exceeds the ready tensor"
    );
    check_hierarchical_epoch(epoch);
    TORCH_CHECK(timeout_ns > 0, "ready timeout must be positive");
    hierarchical_wait_ready_kernel<<<
        1,
        slot_count,
        0,
        at::cuda::getCurrentCUDAStream().stream()
    >>>(
        reinterpret_cast<const uint32_t *>(ready.data_ptr<int>()),
        reinterpret_cast<uint32_t *>(error.data_ptr<int>()),
        first_slot,
        slot_count,
        epoch,
        timeout_ns
    );
    CUDACHECK(cudaGetLastError());
}

void hierarchical_bf16_local_join_entrypoint(
    kittens::py::TKParallelTensor &barrier,
    const at::Tensor &error,
    uint64_t timeout_ns
) {
    c10::cuda::CUDAGuard guard(barrier.data_.device());
    kittens::py::parallel_tensor_check<pk_fused_globals::barrier_pgl>(
        barrier
    );
    TORCH_CHECK(
        barrier.data_.is_cuda()
            && barrier.data_.is_contiguous()
            && barrier.data_.scalar_type() == at::ScalarType::Int,
        "barrier must be a contiguous CUDA INT32 tensor"
    );
    TORCH_CHECK(
        barrier.data_.dim() == 3
            && barrier.data_.size(0) >= 2
            && barrier.data_.size(1) >= 1
            && barrier.data_.size(2) >= 13,
        "barrier is too small for the hierarchical completion counter"
    );
    TORCH_CHECK(
        barrier.local_rank_ >= 0
            && barrier.local_rank_ < pk_fused_globals::NUM_DEVICES,
        "hierarchical local rank must be in the eight-GPU NVLS group"
    );
    check_trace_tensor(error, at::ScalarType::Int, 1, "error");
    kittens::py::device_check(barrier.data_, error);
    TORCH_CHECK(timeout_ns > 0, "local join timeout must be positive");
    auto barrier_pgl = kittens::py::parallel_tensor_to_pgl<
        pk_fused_globals::barrier_pgl
    >(barrier);
    hierarchical_local_join_kernel<<<
        1, 1, 0, at::cuda::getCurrentCUDAStream().stream()
    >>>(
        barrier_pgl,
        reinterpret_cast<uint32_t *>(error.data_ptr<int>()),
        barrier.local_rank_,
        timeout_ns
    );
    CUDACHECK(cudaGetLastError());
}

void fp4_entrypoint(
    const at::Tensor &A,
    const at::Tensor &B,
    kittens::py::TKParallelTensor &C,
    kittens::py::TKParallelTensor &data,
    kittens::py::TKParallelTensor &scales,
    kittens::py::TKParallelTensor &barrier,
    const std::string &mode_string,
    const std::string &fp4_mode_string,
    int num_comp_ctas,
    int num_comm_ctas,
    int comm_threads,
    bool instrument,
    const at::Tensor &compute_trace,
    const at::Tensor &communication_trace,
    const at::Tensor &compute_smids,
    const at::Tensor &communication_smids
) {
    c10::cuda::CUDAGuard guard(A.device());
    auto mode = parse_mode(mode_string);
    Fp4Mode fp4_mode;
    if (fp4_mode_string == "pre") fp4_mode = Fp4Mode::PRE;
    else if (fp4_mode_string == "post") fp4_mode = Fp4Mode::POST;
    else TORCH_CHECK(false, "fp4_mode must be pre or post");
    TORCH_CHECK(mode != LaunchMode::TWO_STREAM, "FP4 path currently uses one stream");

    if (instrument) {
        check_trace_tensor(
            compute_trace, at::ScalarType::Long,
            num_comp_ctas * TRACE_POINTS, "compute_trace"
        );
        check_trace_tensor(
            communication_trace, at::ScalarType::Long,
            num_comm_ctas * TRACE_POINTS, "communication_trace"
        );
        check_trace_tensor(
            compute_smids, at::ScalarType::Int,
            num_comp_ctas, "compute_smids"
        );
        check_trace_tensor(
            communication_smids, at::ScalarType::Int,
            num_comm_ctas, "communication_smids"
        );
    }

    auto G = make_fp4_globals(
        A, B, C, data, scales, barrier, num_comp_ctas, num_comm_ctas
    );
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto *compute_trace_ptr = instrument
        ? compute_trace.data_ptr<int64_t>() : nullptr;
    auto *communication_trace_ptr = instrument
        ? communication_trace.data_ptr<int64_t>() : nullptr;
    auto *compute_smids_ptr = instrument
        ? compute_smids.data_ptr<int>() : nullptr;
    auto *communication_smids_ptr = instrument
        ? communication_smids.data_ptr<int>() : nullptr;

    if (fp4_mode == Fp4Mode::PRE) {
        launch_fp4_pre_compute(
            G, num_comp_ctas, mode != LaunchMode::DEFAULT_STREAM,
            mode == LaunchMode::DEFAULT_STREAM || mode == LaunchMode::PDL_GRID,
            instrument, stream, compute_trace_ptr, compute_smids_ptr
        );
    } else {
        launch_fp4_post_compute(
            G, num_comp_ctas,
            mode != LaunchMode::DEFAULT_STREAM,
            mode == LaunchMode::DEFAULT_STREAM || mode == LaunchMode::PDL_GRID,
            instrument, stream, compute_trace_ptr, compute_smids_ptr
        );
    }

    // Keep the dependency semantics matched to the BF16 split path.  Ordinary
    // completion uses the eight-rank grid-ready gate, whole-grid PDL adds the
    // same-device programmatic wait, and tile PDL consumes per-tile counters.
    WaitMode wait_mode = WaitMode::GRID_COUNTER;
    bool enable_pdl = mode == LaunchMode::PDL_GRID || mode == LaunchMode::PDL_TILE;
    if (mode == LaunchMode::PDL_GRID) wait_mode = WaitMode::PDL_GRID;
    else if (mode == LaunchMode::PDL_TILE) wait_mode = WaitMode::TILE_COUNTER;
    if (fp4_mode == Fp4Mode::PRE) {
        launch_fp4_communication<Fp4Mode::PRE>(
            G, num_comm_ctas, comm_threads, wait_mode, enable_pdl,
            instrument, stream, communication_trace_ptr, communication_smids_ptr
        );
    } else {
        launch_fp4_communication<Fp4Mode::POST>(
            G, num_comm_ctas, comm_threads, wait_mode, enable_pdl,
            instrument, stream, communication_trace_ptr, communication_smids_ptr
        );
    }
    launch_fp4_reset(G, stream);
}

void fp4_pre_producer_only_entrypoint(
    const at::Tensor &A,
    const at::Tensor &B,
    kittens::py::TKParallelTensor &C,
    kittens::py::TKParallelTensor &data,
    kittens::py::TKParallelTensor &scales,
    kittens::py::TKParallelTensor &barrier,
    int num_comp_ctas
) {
    c10::cuda::CUDAGuard guard(A.device());
    auto G = make_fp4_globals(
        A, B, C, data, scales, barrier, num_comp_ctas, 1
    );
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_fp4_pre_compute(
        G, num_comp_ctas, false, false, false, stream, nullptr, nullptr
    );
}

void fp4_pre_communication_only_entrypoint(
    const at::Tensor &A,
    const at::Tensor &B,
    kittens::py::TKParallelTensor &C,
    kittens::py::TKParallelTensor &data,
    kittens::py::TKParallelTensor &scales,
    kittens::py::TKParallelTensor &barrier,
    int num_comm_ctas,
    int comm_threads
) {
    c10::cuda::CUDAGuard guard(A.device());
    auto G = make_fp4_globals(
        A, B, C, data, scales, barrier, 1, num_comm_ctas
    );
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_fp4_communication<Fp4Mode::PRE>(
        G, num_comm_ctas, comm_threads, WaitMode::NONE, false,
        false, stream, nullptr, nullptr
    );
}

void fp4_pre_external_pdl_communication_only_entrypoint(
    const at::Tensor &A,
    const at::Tensor &B,
    kittens::py::TKParallelTensor &C,
    kittens::py::TKParallelTensor &data,
    kittens::py::TKParallelTensor &scales,
    kittens::py::TKParallelTensor &barrier,
    int num_comm_ctas,
    int comm_threads,
    bool enable_pdl
) {
    c10::cuda::CUDAGuard guard(A.device());
    auto G = make_fp4_globals(
        A, B, C, data, scales, barrier, 1, num_comm_ctas
    );
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_fp4_communication<Fp4Mode::PRE>(
        G, num_comm_ctas, comm_threads,
        WaitMode::TILE_COUNTER_PER_RANK, enable_pdl,
        false, stream, nullptr, nullptr
    );
    launch_fp4_reset(G, stream);
}

void fp4_reset_entrypoint(
    const at::Tensor &A,
    const at::Tensor &B,
    kittens::py::TKParallelTensor &C,
    kittens::py::TKParallelTensor &data,
    kittens::py::TKParallelTensor &scales,
    kittens::py::TKParallelTensor &barrier
) {
    c10::cuda::CUDAGuard guard(A.device());
    auto G = make_fp4_globals(A, B, C, data, scales, barrier, 1, 1);
    launch_fp4_reset(G, at::cuda::getCurrentCUDAStream().stream());
}

void interference_entrypoint(
    const at::Tensor &A,
    const at::Tensor &B,
    kittens::py::TKParallelTensor &compute_C,
    kittens::py::TKParallelTensor &compute_barrier,
    kittens::py::TKParallelTensor &communication_C,
    kittens::py::TKParallelTensor &communication_barrier,
    int num_comp_ctas,
    int num_comm_ctas,
    int comm_threads,
    int comm_smem_bytes,
    bool instrument,
    const at::Tensor &compute_trace,
    const at::Tensor &communication_trace,
    const at::Tensor &compute_smids,
    const at::Tensor &communication_smids
) {
    c10::cuda::CUDAGuard guard(A.device());
    TORCH_CHECK(comm_smem_bytes >= 0 && comm_smem_bytes <= MAX_SHARED_MEMORY);
    TORCH_CHECK(
        num_comp_ctas + num_comm_ctas <= NUM_SMS,
        "interference compute and communication CTA roles exceed H100 SMs"
    );
    if (instrument) {
        check_trace_tensor(
            compute_trace, at::ScalarType::Long,
            num_comp_ctas * TRACE_POINTS, "compute_trace"
        );
        check_trace_tensor(
            communication_trace, at::ScalarType::Long,
            num_comm_ctas * TRACE_POINTS, "communication_trace"
        );
        check_trace_tensor(
            compute_smids, at::ScalarType::Int,
            num_comp_ctas, "compute_smids"
        );
        check_trace_tensor(
            communication_smids, at::ScalarType::Int,
            num_comm_ctas, "communication_smids"
        );
    }

    auto compute_G = make_globals(
        A, B, compute_C, compute_barrier, num_comp_ctas, num_comm_ctas
    );
    auto communication_G = make_globals(
        A, B, communication_C, communication_barrier, 1, num_comm_ctas
    );
    auto main_stream = at::cuda::getCurrentCUDAStream().stream();
    auto &state = stream_state();
    auto *compute_trace_ptr = instrument
        ? compute_trace.data_ptr<int64_t>() : nullptr;
    auto *communication_trace_ptr = instrument
        ? communication_trace.data_ptr<int64_t>() : nullptr;
    auto *compute_smids_ptr = instrument
        ? compute_smids.data_ptr<int>() : nullptr;
    auto *communication_smids_ptr = instrument
        ? communication_smids.data_ptr<int>() : nullptr;

    // This is a mechanism-only calibration: GEMM and AllReduce use distinct
    // output tensors, so the communication grid has no data dependency on the
    // compute grid.  Their CTA shapes and launch counts are independently
    // controlled while the total one-CTA/SM role count is capped at NUM_SMS.
    CUDACHECK(cudaEventRecord(state.launch_ready, main_stream));
    CUDACHECK(cudaStreamWaitEvent(
        state.communication, state.launch_ready, 0
    ));
    launch_compute(
        compute_G, num_comp_ctas, false, false, instrument, main_stream,
        compute_trace_ptr, compute_smids_ptr
    );
    launch_communication(
        communication_G, num_comm_ctas, comm_threads, comm_smem_bytes,
        WaitMode::NONE, false, instrument, state.communication, nullptr,
        communication_trace_ptr, communication_smids_ptr
    );
    CUDACHECK(cudaEventRecord(
        state.communication_done, state.communication
    ));
    CUDACHECK(cudaStreamWaitEvent(
        main_stream, state.communication_done, 0
    ));
    launch_split_reset(compute_G, main_stream);
}

void communication_only_entrypoint(
    const at::Tensor &A,
    const at::Tensor &B,
    kittens::py::TKParallelTensor &C,
    kittens::py::TKParallelTensor &barrier,
    int num_comm_ctas,
    int comm_threads,
    int comm_smem_bytes,
    bool instrument,
    const at::Tensor &communication_trace,
    const at::Tensor &communication_smids
) {
    c10::cuda::CUDAGuard guard(A.device());
    TORCH_CHECK(comm_smem_bytes >= 0 && comm_smem_bytes <= MAX_SHARED_MEMORY);
    if (instrument) {
        check_trace_tensor(
            communication_trace, at::ScalarType::Long,
            num_comm_ctas * TRACE_POINTS, "communication_trace"
        );
        check_trace_tensor(
            communication_smids, at::ScalarType::Int,
            num_comm_ctas, "communication_smids"
        );
    }
    auto G = make_globals(A, B, C, barrier, 1, num_comm_ctas);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_communication(
        G, num_comm_ctas, comm_threads, comm_smem_bytes,
        WaitMode::NONE, false, instrument, stream, nullptr,
        instrument ? communication_trace.data_ptr<int64_t>() : nullptr,
        instrument ? communication_smids.data_ptr<int>() : nullptr
    );
}

template <int NUM_THREADS>
pybind11::dict communication_attributes(int dynamic_smem) {
    cudaFuncAttributes attributes;
    auto kernel = split_communication_kernel<NUM_THREADS, false>;
    CUDACHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_smem
    ));
    CUDACHECK(cudaFuncGetAttributes(&attributes, kernel));
    int occupancy = 0;
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, kernel, NUM_THREADS, dynamic_smem
    ));
    pybind11::dict result;
    result["threads"] = NUM_THREADS;
    result["num_regs"] = attributes.numRegs;
    result["static_smem_bytes"] = attributes.sharedSizeBytes;
    result["max_dynamic_smem_bytes"] = attributes.maxDynamicSharedSizeBytes;
    result["requested_dynamic_smem_bytes"] = dynamic_smem;
    result["max_threads_per_block"] = attributes.maxThreadsPerBlock;
    result["predicted_blocks_per_sm"] = occupancy;
    return result;
}

template <int NUM_THREADS, Fp4Mode MODE>
pybind11::dict fp4_communication_attributes() {
    cudaFuncAttributes attributes;
    auto kernel = fp4_communication_kernel<NUM_THREADS, false, MODE>;
    CUDACHECK(cudaFuncGetAttributes(&attributes, kernel));
    int occupancy = 0;
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, kernel, NUM_THREADS, 0
    ));
    pybind11::dict result;
    result["threads"] = NUM_THREADS;
    result["num_regs"] = attributes.numRegs;
    result["static_smem_bytes"] = attributes.sharedSizeBytes;
    result["max_dynamic_smem_bytes"] = attributes.maxDynamicSharedSizeBytes;
    result["requested_dynamic_smem_bytes"] = 0;
    result["max_threads_per_block"] = attributes.maxThreadsPerBlock;
    result["predicted_blocks_per_sm"] = occupancy;
    result["fp4_mode"] = MODE == Fp4Mode::PRE ? "pre" : "post";
    return result;
}

template <Fp4Mode MODE>
pybind11::dict fp4_resource_report(int comm_threads) {
    auto compute_attributes_for = [](auto kernel) {
        cudaFuncAttributes attributes;
        CUDACHECK(cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            FP4_DYNAMIC_SHARED_MEMORY
        ));
        CUDACHECK(cudaFuncGetAttributes(&attributes, kernel));
        int occupancy = 0;
        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &occupancy,
            kernel,
            pk_fused_config::NUM_THREADS,
            FP4_DYNAMIC_SHARED_MEMORY
        ));
        pybind11::dict result;
        result["threads"] = pk_fused_config::NUM_THREADS;
        result["num_regs"] = attributes.numRegs;
        result["static_smem_bytes"] = attributes.sharedSizeBytes;
        result["dynamic_smem_bytes"] = FP4_DYNAMIC_SHARED_MEMORY;
        result["max_dynamic_smem_bytes"] =
            attributes.maxDynamicSharedSizeBytes;
        result["predicted_blocks_per_sm"] = occupancy;
        return result;
    };

    pybind11::dict compute;
    if constexpr (MODE == Fp4Mode::PRE) {
        compute = compute_attributes_for(fp4_pre_compute_kernel<false>);
    } else {
        compute = compute_attributes_for(fp4_post_compute_kernel<false>);
    }
    compute["fp4_mode"] = MODE == Fp4Mode::PRE ? "pre" : "post";

    pybind11::dict communication;
    switch (comm_threads) {
        case 256:
            communication = fp4_communication_attributes<256, MODE>();
            break;
        case 384:
            communication = fp4_communication_attributes<384, MODE>();
            break;
        case 512:
            communication = fp4_communication_attributes<512, MODE>();
            break;
        case 768:
            communication = fp4_communication_attributes<768, MODE>();
            break;
        case 1024:
            communication = fp4_communication_attributes<1024, MODE>();
            break;
        default:
            TORCH_CHECK(false, "unsupported FP4 communication thread count");
    }
    pybind11::dict result;
    result["compute"] = compute;
    result["communication"] = communication;
    result["num_sms"] = NUM_SMS;
    result["tile_m"] = fp4_globals::ROW_BLOCK;
    result["tile_n"] = fp4_globals::COL_BLOCK;
    result["tile_k"] = fp4_globals::RED_BLOCK;
    result["pipeline_stages"] = fp4_globals::PIPELINE_STAGES;
    return result;
}

pybind11::dict fp4_resource_report_dispatch(
    const std::string &fp4_mode, int comm_threads
) {
    if (fp4_mode == "pre") return fp4_resource_report<Fp4Mode::PRE>(comm_threads);
    if (fp4_mode == "post") return fp4_resource_report<Fp4Mode::POST>(comm_threads);
    TORCH_CHECK(false, "fp4_mode must be pre or post");
}

pybind11::dict resource_report(int comm_threads, int comm_smem_bytes) {
    cudaFuncAttributes compute_attributes;
    CUDACHECK(cudaFuncSetAttribute(
        split_compute_kernel<false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        pk_fused_config::DYNAMIC_SHARED_MEMORY
    ));
    CUDACHECK(cudaFuncGetAttributes(
        &compute_attributes, split_compute_kernel<false>
    ));
    int compute_occupancy = 0;
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &compute_occupancy,
        split_compute_kernel<false>,
        pk_fused_config::NUM_THREADS,
        pk_fused_config::DYNAMIC_SHARED_MEMORY
    ));

    pybind11::dict compute;
    compute["threads"] = pk_fused_config::NUM_THREADS;
    compute["num_regs"] = compute_attributes.numRegs;
    compute["static_smem_bytes"] = compute_attributes.sharedSizeBytes;
    compute["dynamic_smem_bytes"] = pk_fused_config::DYNAMIC_SHARED_MEMORY;
    compute["max_dynamic_smem_bytes"] =
        compute_attributes.maxDynamicSharedSizeBytes;
    compute["predicted_blocks_per_sm"] = compute_occupancy;
    compute["producer_register_target"] =
        pk_fused_config::PRODUCER_REGISTERS;
    compute["consumer_register_target"] =
        pk_fused_config::CONSUMER_REGISTERS;

    pybind11::dict communication;
    switch (comm_threads) {
        case 256:
            communication = communication_attributes<256>(comm_smem_bytes);
            break;
        case 384:
            communication = communication_attributes<384>(comm_smem_bytes);
            break;
        case 512:
            communication = communication_attributes<512>(comm_smem_bytes);
            break;
        case 768:
            communication = communication_attributes<768>(comm_smem_bytes);
            break;
        case 1024:
            communication = communication_attributes<1024>(comm_smem_bytes);
            break;
        default:
            TORCH_CHECK(false, "unsupported comm_threads");
    }

    pybind11::dict result;
    result["compute"] = compute;
    result["communication"] = communication;
    result["num_sms"] = NUM_SMS;
    result["tile_m"] = pk_fused_globals::ROW_BLOCK;
    result["tile_n"] = pk_fused_globals::COL_BLOCK;
    result["tile_k"] = pk_fused_globals::RED_BLOCK;
    result["pipeline_stages"] = pk_fused_globals::PIPELINE_STAGES;
    return result;
}

} // namespace pdl_gemm_ar

PYBIND11_MODULE(_C, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def(
        "matmul_all_reduce_fused",
        &pk_fused_entrypoint,
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("barrier"),
        pybind11::arg("num_comm_sms")
    );
    m.def(
        "matmul_all_reduce_split",
        &pdl_gemm_ar::split_entrypoint,
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("barrier"),
        pybind11::arg("mode"),
        pybind11::arg("num_comp_ctas"),
        pybind11::arg("num_comm_ctas"),
        pybind11::arg("comm_threads"),
        pybind11::arg("comm_smem_bytes"),
        pybind11::arg("instrument"),
        pybind11::arg("compute_trace"),
        pybind11::arg("communication_trace"),
        pybind11::arg("compute_smids"),
        pybind11::arg("communication_smids")
    );
    m.def(
        "matmul_local_bf16",
        &pdl_gemm_ar::local_bf16_entrypoint,
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("barrier"),
        pybind11::arg("num_comp_ctas")
    );
    m.def(
        "hierarchical_bf16_producer",
        &pdl_gemm_ar::hierarchical_bf16_producer_entrypoint,
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("barrier"),
        pybind11::arg("wire"),
        pybind11::arg("ready"),
        pybind11::arg("error"),
        pybind11::arg("mode"),
        pybind11::arg("num_comp_ctas"),
        pybind11::arg("num_pack_ctas"),
        pybind11::arg("pack_threads"),
        pybind11::arg("epoch")
    );
    m.def(
        "hierarchical_bf16_unpack",
        &pdl_gemm_ar::hierarchical_bf16_unpack_entrypoint,
        pybind11::arg("C"),
        pybind11::arg("wire"),
        pybind11::arg("first_slot"),
        pybind11::arg("slot_count"),
        pybind11::arg("unpack_threads")
    );
    m.def(
        "hierarchical_bf16_pack_only",
        &pdl_gemm_ar::hierarchical_bf16_pack_only_entrypoint,
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("barrier"),
        pybind11::arg("wire"),
        pybind11::arg("ready"),
        pybind11::arg("error"),
        pybind11::arg("num_pack_ctas"),
        pybind11::arg("pack_threads"),
        pybind11::arg("epoch")
    );
    m.def(
        "hierarchical_wait_ready",
        &pdl_gemm_ar::hierarchical_wait_ready_entrypoint,
        pybind11::arg("ready"),
        pybind11::arg("error"),
        pybind11::arg("first_slot"),
        pybind11::arg("slot_count"),
        pybind11::arg("epoch"),
        pybind11::arg("timeout_ns")
    );
    m.def(
        "hierarchical_bf16_local_join",
        &pdl_gemm_ar::hierarchical_bf16_local_join_entrypoint,
        pybind11::arg("barrier"),
        pybind11::arg("error"),
        pybind11::arg("timeout_ns")
    );
    m.def(
        "matmul_all_reduce_fp4",
        &pdl_gemm_ar::fp4_entrypoint,
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("data"),
        pybind11::arg("scales"),
        pybind11::arg("barrier"),
        pybind11::arg("mode"),
        pybind11::arg("fp4_mode"),
        pybind11::arg("num_comp_ctas"),
        pybind11::arg("num_comm_ctas"),
        pybind11::arg("comm_threads"),
        pybind11::arg("instrument"),
        pybind11::arg("compute_trace"),
        pybind11::arg("communication_trace"),
        pybind11::arg("compute_smids"),
        pybind11::arg("communication_smids")
    );
    m.def(
        "fp4_pre_producer_only",
        &pdl_gemm_ar::fp4_pre_producer_only_entrypoint,
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("data"),
        pybind11::arg("scales"),
        pybind11::arg("barrier"),
        pybind11::arg("num_comp_ctas")
    );
    m.def(
        "fp4_pre_communication_only",
        &pdl_gemm_ar::fp4_pre_communication_only_entrypoint,
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("data"),
        pybind11::arg("scales"),
        pybind11::arg("barrier"),
        pybind11::arg("num_comm_ctas"),
        pybind11::arg("comm_threads")
    );
    m.def(
        "fp4_pre_external_pdl_communication_only",
        &pdl_gemm_ar::fp4_pre_external_pdl_communication_only_entrypoint,
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("data"),
        pybind11::arg("scales"),
        pybind11::arg("barrier"),
        pybind11::arg("num_comm_ctas"),
        pybind11::arg("comm_threads"),
        pybind11::arg("enable_pdl") = true
    );
    m.def(
        "fp4_reset",
        &pdl_gemm_ar::fp4_reset_entrypoint,
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("data"),
        pybind11::arg("scales"),
        pybind11::arg("barrier")
    );
    m.def(
        "all_reduce_split",
        &pdl_gemm_ar::communication_only_entrypoint,
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("barrier"),
        pybind11::arg("num_comm_ctas"),
        pybind11::arg("comm_threads"),
        pybind11::arg("comm_smem_bytes"),
        pybind11::arg("instrument"),
        pybind11::arg("communication_trace"),
        pybind11::arg("communication_smids")
    );
    m.def(
        "matmul_all_reduce_interference",
        &pdl_gemm_ar::interference_entrypoint,
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("compute_C"),
        pybind11::arg("compute_barrier"),
        pybind11::arg("communication_C"),
        pybind11::arg("communication_barrier"),
        pybind11::arg("num_comp_ctas"),
        pybind11::arg("num_comm_ctas"),
        pybind11::arg("comm_threads"),
        pybind11::arg("comm_smem_bytes"),
        pybind11::arg("instrument"),
        pybind11::arg("compute_trace"),
        pybind11::arg("communication_trace"),
        pybind11::arg("compute_smids"),
        pybind11::arg("communication_smids")
    );
    m.def(
        "resource_report",
        &pdl_gemm_ar::resource_report,
        pybind11::arg("comm_threads"),
        pybind11::arg("comm_smem_bytes")
    );
    m.def(
        "fp4_resource_report",
        &pdl_gemm_ar::fp4_resource_report_dispatch,
        pybind11::arg("fp4_mode"),
        pybind11::arg("comm_threads")
    );
}
