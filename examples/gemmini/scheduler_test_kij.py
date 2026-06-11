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

    # the values of a and b must be the same, cause we have one single load pattern
    def get_load_b_pat(self):
        return ("k", "j"), (1, 4, 16, 16)

    def get_init_out_pat(self):
        return ("i", "j"), (16, 16)

    def get_gemm_pat(self):
        return ("i", "j", "k"), (16, 16, 16)

    def get_store_pat(self):
        return ("i", "j"), (16, 16)


def generate_sample4_schedule():
    K = 1024
    M = 1024
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

    order = ("koo", "joo", "io", "koi", "joi", "ii", "ji", "ki")
    loads = (2, 1, 2)
    ii = (64, 16)
    jj = (4, 16, 16)
    kk = (4, 16, 16)
    workload_size = (N, M, K)
    mapping = MatMulMapping(order, loads, ii, jj, kk, workload_size)
    sch = GemminiScheduler(mapping)

    gemmini = sch.apply_tiling(gemmini)
    # print("after complete_tiling")
    # print(gemmini)
    init_out_t = (ld_acc_i32_vector, ld_acc_i32_vector_v2)
    gemmini = sch.replace_init_out(gemmini, accum_size, init_out_t, GEMM_ACCUM)
    # print("after replace_init_out")
    # print(gemmini)

    gemmini = sch.apply_mapping_order(gemmini)
    # print("after apply_mapping_order")
    # print(gemmini)

    a_instr_t = (ld_i8_block_id1, ld_i8_block_id1_v2)
    b_instr_t = (ld_i8_block_id2, ld_i8_block_id2_v2)
    gemmini = sch.replace_load_buffers(
        gemmini, GEMM_SCRATCH, GEMM_SCRATCH, a_instr_t, b_instr_t
    )
    # print("after replace_load_buffers")
    # print(gemmini)

    gemm_instr_t = (matmul_acc_i8, matmul_acc_i8_v2)
    gemmini = sch.replace_gemm(gemmini, gemm_instr_t)
    print("after replace_gemm")
    print(gemmini)

    store_instr_t = (st_acc_i8, st_acc_i8_v2)
    gemmini = sch.replace_store(gemmini, store_instr_t)
    print("after replace_store")
    print(gemmini)
    gemmini = fuse_all_loops(gemmini, gemmini.body()[0])
    gemmini = simplify(gemmini)

    print(gemmini)
    print(gemmini.c_code_str())


generate_sample4_schedule()
