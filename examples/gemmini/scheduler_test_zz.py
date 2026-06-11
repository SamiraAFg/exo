from __future__ import annotations

import os
import sys

from exo.libs.memories import GEMM_SCRATCH, GEMM_ACCUM
from exo import proc, instr, DRAM, config, ExoType
from exo.stdlib.scheduling import *
from exo.stdlib.stdlib import *
from exo.stdlib.inspection import *
from exo.platforms.gemmini import (
    ld_i8_block_id1,
    ld_i8_block_id1_v2,
    # ld_i8_block_id1_conv,
    # ld_i8_block_id1_v2_conv,
    ld_i8_block_id2,
    ld_i8_block_id2_v2,
    ld_acc_i32_vector,
    ld_acc_i32_vector_v2,
    zero_acc_i32,
    zero_acc_i32_v2,
    matmul_acc_i8,
    matmul_acc_i8_v2,
    st_acc_i8,
    st_acc_i8_v2,
    acc_scale,
    clamp,
)
from gemmini_schedules import *

from scheduler import Scheduler, MatMulMapping

ld_i8_block_id1 = reorder_loops(ld_i8_block_id1, "i j")
ld_i8_block_id2 = reorder_loops(ld_i8_block_id2, "i j")


def matmul_algorithm():
    @proc
    def matmul(
        N: size,
        M: size,
        K: size,
        scale: f32,
        act: bool,
        A: i8[N, K] @ DRAM,
        B: i8[K, M] @ DRAM,
        D: i32[1, M] @ DRAM,
        C: i8[N, M] @ DRAM,
    ):

        # Algorithm starts here
        for i in seq(0, N):
            for j in seq(0, M):
                res: i32 @ DRAM
                res = D[0, j]
                for k in seq(0, K):
                    a2: i32
                    b2: i32
                    a2 = A[i, k]
                    b2 = B[k, j]
                    res += a2 * b2

                src_tmp: i32
                src_tmp = res
                tmp_res1: f32
                acc_scale(src_tmp, tmp_res1, scale)
                tmp_res2: i8
                clamp(tmp_res1, tmp_res2)
                if act == True:
                    tmp_res2 = relu(tmp_res2)
                C[i, j] = tmp_res2
        # Algorithm ends here. 23 lines excluding newlines

    return matmul


class GemminiScheduler(Scheduler):
    def get_load_a_pat(self):
        return ("i", "k"), (1, 4, 16, 16)

    def get_load_b_pat(self):
        return ("k", "j"), (1, 4, 16, 16)

    def get_init_out_pat(self):
        return ("i", "j"), (16, 16)

    def get_gemm_pat(self):
        return ("i", "j", "k"), (16, 16, 16)

    def get_store_pat(self):
        return ("i", "j"), (16, 16)


def generate_sample3_schedule():
    K = 512
    M = 1024
    N = 256
    cpu = rename(
        matmul_algorithm(), "matmul_on_cpu"
    )  # Rename "matmul" to "matmul_on_cpu"
    cpu = cpu.partial_eval(N=N, M=M, K=K)
    # cpu = cpu.add_assertion("N % 256 == 0")
    # cpu = cpu.add_assertion("M % 256 == 0")

    # Rename the procedure to "matmul_on_gemmini"
    gemmini = rename(cpu, "matmul_on_gemmini")

    print("")
    print("===== THIS IS THE ORIGINAL MATMUL ALGORITHM BEFORE SCHEDULING ====")
    print(gemmini)
    print("===== THIS IS THE ORIGINAL MATMUL ALGORITHM BEFORE SCHEDULING ====")
    print("")

    # Parameters
    accum_size = 16 * 1024

    sc_size = 256 * 1024

    order = ("joo", "ko", "joi", "io", "ii", "ji", "ki")
    loads = (-1, 0, 0)
    ii = (16, 16)
    jj = (16, 4, 16)
    kk = (32, 16)
    workload_size = (N, M, K)
    mapping = MatMulMapping(order, loads, ii, jj, kk, workload_size)
    sch = GemminiScheduler(mapping)

    gemmini = sch.multi_loop_tiling(gemmini)
    print("after complete_tiling")
    print(gemmini)
    init_out_t = (ld_acc_i32_vector, ld_acc_i32_vector_v2)
    gemmini = sch.replace_init_out(gemmini, accum_size, init_out_t, GEMM_ACCUM)
    print("after replace_init_out")
    print(gemmini)

    gemmini = sch.apply_mapping_order(gemmini)
    print("after apply_mapping_order")
    print(gemmini)

    a_instr_t = (ld_i8_block_id1, ld_i8_block_id1_v2)
    b_instr_t = (ld_i8_block_id2, ld_i8_block_id2_v2)
    gemmini = sch.replace_load_buffers(
        gemmini, GEMM_SCRATCH, GEMM_SCRATCH, a_instr_t, b_instr_t, sc_size
    )
    print("after replace_load_buffers")
    print(gemmini)

    gemm_instr_t = (matmul_acc_i8, matmul_acc_i8_v2)
    gemmini = sch.replace_gemm(gemmini, gemm_instr_t)
    # print("after replace_gemm")
    # print(gemmini)

    store_instr_t = (st_acc_i8, st_acc_i8_v2)
    gemmini = sch.replace_store(gemmini, store_instr_t)
    print("after replace_store")
    print(gemmini)
    gemmini = fuse_all_loops(gemmini, gemmini.body()[0])
    gemmini = simplify(gemmini)

    print(gemmini)
    print(gemmini.c_code_str())
    # the problem is that the offset of A_temp and B_temp are not calculated correctly, as the stride of the last index
    # is 1 always, this means that they consider the allocated spaces segmented [_, _, 16, 16] the last one has to be 16!
    # you also can't make the last one 64 becasue then it conflicts with gemm pattern! So somehow you have to make the loads like
    # 4, 16, 16!! you have to rewrite your approach for replace_load_buffers


def generate_sample2_schedule():
    K = 512
    M = 512
    N = 512
    cpu = rename(
        matmul_algorithm(), "matmul_on_cpu"
    )  # Rename "matmul" to "matmul_on_cpu"
    cpu = cpu.partial_eval(N=N, M=M, K=K)
    # cpu = cpu.add_assertion("N % 256 == 0")
    # cpu = cpu.add_assertion("M % 256 == 0")

    # Rename the procedure to "matmul_on_gemmini"
    gemmini = rename(cpu, "matmul_on_gemmini")

    print("")
    print("===== THIS IS THE ORIGINAL MATMUL ALGORITHM BEFORE SCHEDULING ====")
    print(gemmini)
    print("===== THIS IS THE ORIGINAL MATMUL ALGORITHM BEFORE SCHEDULING ====")
    print("")

    # Parameters
    accum_size = 16 * 1024

    sc_size = 256 * 1024

    order = ("joo", "io", "ko", "joi", "ii", "ji", "ki")
    loads = (1, 0, 1)
    ii = (32, 16)
    jj = (4, 8, 16)
    kk = (32, 16)
    workload_size = (N, M, K)
    mapping = MatMulMapping(order, loads, ii, jj, kk, workload_size)
    sch = GemminiScheduler(mapping)

    gemmini = sch.multi_loop_tiling(gemmini)
    # print("after complete_tiling")
    # print(gemmini)
    init_out_t = (ld_acc_i32_vector, ld_acc_i32_vector_v2)
    gemmini = sch.replace_init_out(gemmini, accum_size, init_out_t, GEMM_ACCUM)
    # print("after replace_init_out")
    # print(gemmini)

    gemmini = sch.apply_mapping_order(gemmini)
    print("after apply_mapping_order")
    print(gemmini)

    a_instr_t = (ld_i8_block_id1, ld_i8_block_id1_v2)
    b_instr_t = (ld_i8_block_id2, ld_i8_block_id2_v2)
    gemmini = sch.replace_load_buffers(
        gemmini, GEMM_SCRATCH, GEMM_SCRATCH, a_instr_t, b_instr_t, sc_size
    )
    print("after replace_load_buffers")
    print(gemmini)

    gemm_instr_t = (matmul_acc_i8, matmul_acc_i8_v2)
    gemmini = sch.replace_gemm(gemmini, gemm_instr_t)
    # print("after replace_gemm")
    # print(gemmini)

    store_instr_t = (st_acc_i8, st_acc_i8_v2)
    gemmini = sch.replace_store(gemmini, store_instr_t)
    # print("after replace_store")
    # print(gemmini)
    gemmini = fuse_all_loops(gemmini, gemmini.body()[0])
    gemmini = simplify(gemmini)

    print(gemmini)
    print(gemmini.c_code_str())


def generate_sample1_schedule():
    K = 512
    M = 256
    N = 1024
    cpu = rename(
        matmul_algorithm(), "matmul_on_cpu"
    )  # Rename "matmul" to "matmul_on_cpu"
    cpu = cpu.partial_eval(N=N, M=M, K=K)
    # cpu = cpu.add_assertion("N % 256 == 0")
    # cpu = cpu.add_assertion("M % 256 == 0")

    # Rename the procedure to "matmul_on_gemmini"
    gemmini = rename(cpu, "matmul_on_gemmini")

    print("")
    print("===== THIS IS THE ORIGINAL MATMUL ALGORITHM BEFORE SCHEDULING ====")
    print(gemmini)
    print("===== THIS IS THE ORIGINAL MATMUL ALGORITHM BEFORE SCHEDULING ====")
    print("")

    # Parameters
    accum_size = 16 * 1024

    sc_size = 256 * 1024

    order = ("io", "ko", "jo", "ii", "ji", "ki")
    loads = (0, -1, 0)
    ii = (64, 16)
    jj = (16, 16)
    kk = (32, 16)
    workload_size = (N, M, K)
    mapping = MatMulMapping(order, loads, ii, jj, kk, workload_size)
    sch = GemminiScheduler(mapping)

    gemmini = sch.multi_loop_tiling(gemmini)
    # print("after complete_tiling")
    # print(gemmini)
    init_out_t = (ld_acc_i32_vector, ld_acc_i32_vector_v2)
    gemmini = sch.replace_init_out(gemmini, accum_size, init_out_t, GEMM_ACCUM)
    print("after replace_init_out")
    print(gemmini)

    gemmini = sch.apply_mapping_order(gemmini)
    # print("after apply_mapping_order")
    # print(gemmini)

    a_instr_t = (ld_i8_block_id1, ld_i8_block_id1_v2)
    b_instr_t = (ld_i8_block_id2, ld_i8_block_id2_v2)
    gemmini = sch.replace_load_buffers(
        gemmini, GEMM_SCRATCH, GEMM_SCRATCH, a_instr_t, b_instr_t, sc_size
    )
    print("after replace_load_buffers")
    print(gemmini)

    gemm_instr_t = (matmul_acc_i8, matmul_acc_i8_v2)
    gemmini = sch.replace_gemm(gemmini, gemm_instr_t)
    # print("after replace_gemm")
    # print(gemmini)

    store_instr_t = (st_acc_i8, st_acc_i8_v2)
    gemmini = sch.replace_store(gemmini, store_instr_t)
    # print("after replace_store")
    # print(gemmini)
    gemmini = fuse_all_loops(gemmini, gemmini.body()[0])
    gemmini = simplify(gemmini)

    print(gemmini)
    print(gemmini.c_code_str())


# generate_sample1_schedule()
# generate_sample2_schedule()
# generate_sample3_schedule()

# test the generated codes on Spike -> checked! All generated codes work on Spike simulator
@proc
def conv_on_cpu_stride_1(
    batch_size: size,
    out_dim: size,
    out_channel: size,
    kernel_dim: size,
    in_channel: size,
    in_dim: size,
    C: i8[batch_size, out_dim, out_dim, out_channel],
    D: i32[1, out_channel],
    A: i8[batch_size, in_dim, in_dim, in_channel],
    B: i8[kernel_dim, kernel_dim, in_channel, out_channel],
    act: bool,
    scale: f32,
):
    assert out_dim == in_dim - kernel_dim + 1

    for b in seq(0, batch_size):
        for orow in seq(0, out_dim):
            for ocol in seq(0, out_dim):
                for j in seq(0, out_channel):

                    res: i32
                    res = D[0, j]
                    for krow in seq(0, kernel_dim):
                        for kcol in seq(0, kernel_dim):
                            for kch in seq(0, in_channel):
                                # w_s: i8 @ DRAM
                                # w_s = weights[krow, kcol, kch, j]

                                # i_s: i8 @ DRAM
                                # i_s = inp[b, orow + krow, ocol + kcol, kch]

                                a2: i32
                                b2: i32
                                b2 = B[krow, kcol, kch, j]
                                a2 = A[b, orow + krow, ocol + kcol, kch]

                                res += a2 * b2

                    src_tmp: i32
                    src_tmp = res
                    tmp_res1: f32
                    acc_scale(src_tmp, tmp_res1, scale)
                    tmp_res2: i8
                    clamp(tmp_res1, tmp_res2)
                    if act == True:
                        tmp_res2 = relu(tmp_res2)

                    C[b, orow, ocol, j] = tmp_res2


def test_conv_autosch():
    batch_size = 4
    out_channel = 256
    kernel_dim = 4
    in_channel = 32
    in_dim = 19
    out_dim = int((in_dim - kernel_dim) / 1 + 1)
    assert out_dim == 16

    cpu = rename(conv_on_cpu_stride_1, "my_conv_3_cpu")
    cpu = cpu.partial_eval(
        batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim
    )
    conv = rename(cpu, "my_conv_3")

    # Grab cursors
    b_loop = conv.find_loop("b")
    krow_loop = conv.find_loop("krow")
    kcol_loop = conv.find_loop("kcol")
    orow_loop = conv.find_loop("orow")

    # make it a matmul (on-the-fly im2col)
    conv = simplify(mult_loops(conv, orow_loop, "ii"))
    # print(conv)
    conv = simplify(mult_loops(conv, b_loop, "i"))
    conv = simplify(mult_loops(conv, krow_loop, "ko"))
    # print(conv)
    conv = simplify(mult_loops(conv, conv.find_loop("for ko in _:_"), "k"))
    conv = simplify(conv)
    print("")
    print("===== THIS IS THE ORIGINAL MATMUL ALGORITHM BEFORE SCHEDULING ====")
    print(conv)
    print("===== THIS IS THE ORIGINAL MATMUL ALGORITHM BEFORE SCHEDULING ====")
    print("")

    # Parameters
    accum_size = 16 * 1024

    sc_size = 256 * 1024

    order = ("io", "ko", "jo", "ii", "ji", "ki")
    loads = (0, -1, 0)
    ii = (64, 16)
    jj = (16, 16)
    kk = (32, 16)
    workload_size = (N, M, K)
    mapping = MatMulMapping(order, loads, ii, jj, kk, workload_size)
    sch = GemminiScheduler(mapping)

    conv = sch.multi_loop_tiling(conv)
    print("after complete_tiling")
    print(conv)
    init_out_t = (ld_acc_i32_vector, ld_acc_i32_vector_v2)
    conv = sch.replace_init_out(conv, accum_size, init_out_t, GEMM_ACCUM)
    print("after replace_init_out")
    print(conv)

    conv = sch.apply_mapping_order(conv)
    print("after apply_mapping_order")
    print(conv)

    a_instr_t = (ld_i8_block_id1, ld_i8_block_id1_v2)
    # a_instr_t = (ld_i8_block_id1_conv, ld_i8_block_id1_v2_conv)
    b_instr_t = (ld_i8_block_id2, ld_i8_block_id2_v2)
    conv = sch.replace_load_buffers(
        conv, GEMM_SCRATCH, GEMM_SCRATCH, a_instr_t, b_instr_t, sc_size
    )
    print("after replace_load_buffers")
    print(conv)

    # gemm_instr_t = (matmul_acc_i8, matmul_acc_i8_v2)
    # gemmini = sch.replace_gemm(gemmini, gemm_instr_t)
    # # print("after replace_gemm")
    # # print(gemmini)

    # store_instr_t = (st_acc_i8, st_acc_i8_v2)
    # gemmini = sch.replace_store(gemmini, store_instr_t)
    # # print("after replace_store")
    # # print(gemmini)
    # gemmini = fuse_all_loops(gemmini, gemmini.body()[0])
    # gemmini = simplify(gemmini)

    # print(gemmini)
    # print(gemmini.c_code_str())


test_conv_autosch()
# generate_sample1_schedule()
