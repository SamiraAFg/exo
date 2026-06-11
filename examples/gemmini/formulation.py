from __future__ import annotations

from exo.platforms.gemmini import (
    ld_i8_block_id1_new,
    ld_i8_block_id1_v2_new,
    ld_i8_block_id2_new,
    ld_i8_block_id2_v2_new,
    ld_acc_i32_vector,
    ld_acc_i32_vector_v2,
    zero_acc_i32,
    zero_acc_i32_v2,
    ld_i8_block_id1,
    ld_i8_block_id1_v2,
    ld_i8_block_id2,
    ld_i8_block_id2_v2,
    matmul_acc_i8,
    matmul_acc_i8_v2,
    st_acc_i8,
    st_acc_i8_v2,
    acc_scale,
    clamp,
)
from exo.core.LoopIR import LoopIR
from exo.core.LoopIR import T as TT
import exo.API_cursors as pc
from exo.libs.memories import GEMM_SCRATCH, GEMM_ACCUM
from exo import proc, instr, DRAM, config, ExoType
from exo.stdlib.scheduling import *
from exo.stdlib.stdlib import *
from exo.stdlib.inspection import *
import os
import sys


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gemmini_schedules import *

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


def matmul_algorithm2():
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
        res: i32 @ DRAM
        # Algorithm starts here
        for i in seq(0, N):
            for j in seq(0, M):
                for k in seq(0, K):
                    if k == 0:

                        res = D[0, j]
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


def matmul_algorithm3():
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
        res: i32[N, M] @ DRAM
        # Algorithm starts here
        for k in seq(0, K):
            for i in seq(0, N):
                for j in seq(0, M):
                    if k == 0:
                        res[i, j] = D[0, j]
                    a2: i32
                    b2: i32
                    a2 = A[i, k]
                    b2 = B[k, j]
                    res[i, j] += a2 * b2

                    src_tmp: i32
                    src_tmp = res[i, j]
                    tmp_res1: f32
                    acc_scale(src_tmp, tmp_res1, scale)
                    tmp_res2: i8
                    clamp(tmp_res1, tmp_res2)
                    if act == True:
                        tmp_res2 = relu(tmp_res2)
                    C[i, j] = tmp_res2
        # Algorithm ends here. 23 lines excluding newlines

    return matmul


# zz_mapping_sample3 = MatMulMapping(('joo', 'ko', 'joi', 'io', 'io', 'ji', 'ki'), (-1, 0, 0), (16, 16), (16, 4, 16), (32, 16), (256, 1024, 512))
# zz_mapping = [('j', 2, 16), ('k', 1, 32), ('j', 1, 4), ('i', 1, 16), ('i', 0, 16), ('j', 0, 16), ('k', 0, 16)]
# inner_loops = [('i', 0, 16), ('j', 0, 16), ('k', 0, 16)]


def test_matmul_zz_sample3_2():
    KK = 512
    M = 1024
    N = 256
    cpu = rename(
        matmul_algorithm(), "matmul_on_cpu"
    )  # Rename "matmul" to "matmul_on_cpu"
    cpu = cpu.partial_eval(N=N, M=M, K=KK)
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
    # Grab cursors
    i_loop = gemmini.find_loop("i")
    j_loop = gemmini.find_loop("j")
    k_loop = gemmini.find_loop("k")
    res_load = gemmini.find("res = D[_]")
    a_assign = gemmini.find("a2 = A[_]")
    b_assign = gemmini.find("b2 = B[_]")
    res_alloc = res_load.prev()

    # Schedule starts here!!
    # Tile loops for a scratchpad and an accumulator
    # O load (do all tiling for i and j first)
    gemmini = reorder_loops(gemmini, "i j")
    gemmini, _ = tile_loops(gemmini, [(j_loop, 16), (i_loop, 16)], perfect=True)
    gemmini = reorder_loops(gemmini, "ji ii")
    gemmini = divide_loop(gemmini, j_loop, 4, ["joo", "joi"], perfect=True)
    gemmini, _ = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
    gemmini = fission_as_much_as_possible(gemmini, res_load)
    print(gemmini)

    # In and W loads (then do k loop tiling and final reordering)
    gemmini = divide_loop(gemmini, k_loop, 16, ["ko", "ki"], perfect=True)
    gemmini = fission_as_much_as_possible(gemmini, k_loop)
    gemmini = reorder_loops_from_idx_zz(
        gemmini, a_assign, ["ko", "joi", "io", "ii", "ji", "ki"]
    )
    # # Bind and lift scratchpad & accumulator memories
    # gemmini, _ = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
    (gemmini, occ_size), a_load, a_alloc = bind_and_lift(
        gemmini, gemmini.forward(a_assign).rhs(), max_size=sc_size
    )
    # occ_size = 256*512
    # print(occ_size)
    (gemmini, _), b_load, b_alloc = bind_and_lift(
        gemmini, gemmini.forward(b_assign).rhs(), max_size=sc_size - occ_size
    )
    print("after bind and lift b")
    print(gemmini)
    # Fission all the loops
    gemmini = fission_as_much_as_possible(gemmini, a_load)
    gemmini = fission_as_much_as_possible(gemmini, b_load)
    print("8: After fissioning")
    print(gemmini)

    # Fix indexing
    gemmini = rearrange_dim(gemmini, a_alloc, [1, 0, 2, 3])
    gemmini = rearrange_dim(gemmini, b_alloc, [0, 1, 3, 2])
    gemmini = reorder_loops_from_idx(gemmini, a_load)
    gemmini = reorder_loops_from_idx(gemmini, b_load)

    # gemmini = reorder_loops_from_idx(
    #     gemmini, gemmini.forward(b_assign).rhs()
    # )
    # gemmini = reorder_loops(gemmini, gemmini.forward(b_assign).parent().parent())
    gemmini = remove_redundant_loops(
        gemmini, a_load, num=2
    )  # original 2: check for final zizag mapping (fusion)
    gemmini = remove_redundant_loops(gemmini, b_load, num=2)
    ## write sth to check if the third loop is 4 or not
    gemmini = divide_loop(
        gemmini,
        gemmini.forward(a_load).parent().parent().parent(),
        4,
        ["koo", "koi"],
        perfect=True,
    )

    # # print("9: After rearranging dimensions")
    # # print(gemmini)

    # # Replace to gemmini calls, inline to v2, and hoist all the configurations
    gemmini = set_memory(gemmini, res_alloc, GEMM_ACCUM)
    gemmini = set_memory(gemmini, a_alloc, GEMM_SCRATCH)
    gemmini = set_memory(gemmini, b_alloc, GEMM_SCRATCH)
    # print("10: After setting memories")
    # print(gemmini)
    # print(ld_i8_block_id1)
    tuples = [
        (ld_acc_i32_vector, ld_acc_i32_vector_v2),
        (ld_i8_block_id1, ld_i8_block_id1_v2),
        (ld_i8_block_id2, ld_i8_block_id2_v2),
        (matmul_acc_i8, matmul_acc_i8_v2),
        (st_acc_i8, st_acc_i8_v2),
    ]
    for t in tuples:
        gemmini = simplify(replace_and_inline(gemmini, t))
    # print("11: After replacing and inlining")
    print(gemmini)
    gemmini = fuse_all_loops(gemmini, gemmini.body()[0])
    gemmini = simplify(gemmini)
    print("after fusion")
    print(gemmini)
    # print(gemmini.c_code_str())


def test_matmul_zz_2():
    KK = 512
    M = 1024
    N = 256
    cpu = rename(
        matmul_algorithm(), "matmul_on_cpu"
    )  # Rename "matmul" to "matmul_on_cpu"
    cpu = cpu.partial_eval(N=N, M=M, K=KK)
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
    # Grab cursors
    i_loop = gemmini.find_loop("i")
    j_loop = gemmini.find_loop("j")
    k_loop = gemmini.find_loop("k")
    res_load = gemmini.find("res = D[_]")
    a_assign = gemmini.find("a2 = A[_]")
    b_assign = gemmini.find("b2 = B[_]")
    res_alloc = res_load.prev()

    # Schedule starts here!!
    # Tile loops for a scratchpad and an accumulator
    # O load (do all tiling for i and j first)
    gemmini = reorder_loops(gemmini, "i j")
    gemmini, _ = tile_loops(gemmini, [(j_loop, 16), (i_loop, 16)], perfect=True)
    gemmini = reorder_loops(gemmini, "ji ii")
    gemmini = divide_loop(gemmini, k_loop, 16, ["ko", "ki"], perfect=True)

    # gemmini = divide_loop(gemmini, j_loop, 4, ["joo", "joi"], perfect=True)
    gemmini, _ = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
    gemmini = fission_as_much_as_possible(gemmini, res_load)
    gemmini = set_memory(gemmini, res_alloc, GEMM_ACCUM)
    gemmini = simplify(
        replace_and_inline(gemmini, (ld_acc_i32_vector, ld_acc_i32_vector_v2))
    )
    print(gemmini)


def test2():
    KK = 512
    M = 1024
    N = 256
    cpu = rename(
        matmul_algorithm(), "matmul_on_cpu"
    )  # Rename "matmul" to "matmul_on_cpu"
    cpu = cpu.partial_eval(N=N, M=M, K=KK)
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
    # Grab cursors
    i_loop = gemmini.find_loop("i")
    j_loop = gemmini.find_loop("j")
    k_loop = gemmini.find_loop("k")
    res_load = gemmini.find("res = D[_]")
    a_assign = gemmini.find("a2 = A[_]")
    b_assign = gemmini.find("b2 = B[_]")
    res_alloc = res_load.prev()

    # Handling Tilings
    gemmini, [ii, ji] = tile_loops(gemmini, [(i_loop, 16), (j_loop, 16)])

    gemmini, [ki] = tile_loops(gemmini, [(k_loop, 16)])
    gemmini = divide_loop(gemmini, j_loop, 4, ["joo", "joi"], perfect=True)

    # first reordering + replace_init_out
    gemmini = reorder_loops_from_idx_zz(
        gemmini, res_alloc, ["joo", "joi", "io", "ii", "ji"]
    )

    gemmini, _ = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
    gemmini = fission_as_much_as_possible(gemmini, res_load)
    gemmini = set_memory(gemmini, res_alloc, GEMM_ACCUM)
    gemmini = simplify(
        replace_and_inline(gemmini, (ld_acc_i32_vector, ld_acc_i32_vector_v2))
    )

    # print("after replace init out")
    # print(gemmini)
    # complete reordering
    gemmini = fission_as_much_as_possible(gemmini, k_loop)
    gemmini = reorder_loops_from_idx_zz(
        gemmini, a_assign, ["joo", "ko", "joi", "io", "ii", "ji", "ki"]
    )
    # print("after complete reordering")
    # print(gemmini)

    # replace load buffers
    # Step 1: lift allocation
    (gemmini, occ_size), a_load, a_alloc = bind_and_lift(
        gemmini, gemmini.forward(a_assign).rhs(), max_size=sc_size
    )
    (gemmini, _), b_load, b_alloc = bind_and_lift(
        gemmini, gemmini.forward(b_assign).rhs(), max_size=sc_size - occ_size
    )
    # Step 2: fission
    gemmini = fission_as_much_as_possible(gemmini, a_load)
    gemmini = fission_as_much_as_possible(gemmini, b_load)

    # Step 2.5: remove redundant loops
    gemmini = remove_redundant_loops(
        gemmini, a_load, num=2
    )  # original 2: check for final zizag mapping (fusion)
    gemmini = remove_redundant_loops(gemmini, b_load, num=2)

    print(gemmini)
    # Step 3: restructure for pattern matching -> 1. back to the original dimension for each tensor + reordering/multiplying loops accordingly (4 steps),
    # 2. tiling to match the instruction patterns 3. Set the memories + call the replace
    gemmini = rearrange_dim(gemmini, a_alloc, [1, 2, 0, 3])
    gemmini = rearrange_dim(gemmini, b_alloc, [0, 3, 1, 2])
    gemmini = reorder_loops_from_idx(gemmini, a_load)
    gemmini = reorder_loops_from_idx(gemmini, b_load)
    # print(gemmini)

    # gemmini = mult_dim(gemmini, a_alloc, 0, 1)
    # gemmini = mult_dim(gemmini, a_alloc, 1, 2)
    # gemmini = mult_dim(gemmini, b_alloc, 0, 1)
    # gemmini = mult_dim(gemmini, b_alloc, 1, 2)
    # print("aftr mult_dim b")

    # print(gemmini)
    # #idea: multiply the same loops then tile it using tile_loops to match the order and values and then try new intrinsics
    loop_list = get_loops_at_or_above(gemmini.forward(a_load))
    gemmini = mult_loops(gemmini, loop_list[-2], "k_a")
    gemmini = mult_loops(gemmini, loop_list[-4], "i_a")
    gemmini = simplify(gemmini)

    loop_list = get_loops_at_or_above(gemmini.forward(b_load))
    gemmini = mult_loops(gemmini, loop_list[-2], "j_b")
    gemmini = mult_loops(gemmini, loop_list[-4], "k_b")
    gemmini = simplify(gemmini)
    gemmini, _ = tile_loops(
        gemmini, [(gemmini.find_loop("i_a"), 16), (gemmini.find_loop("k_a"), 64)]
    )
    gemmini, _ = tile_loops(
        gemmini, [(gemmini.find_loop("k_b"), 16), (gemmini.find_loop("j_b"), 64)]
    )
    # print(gemmini)
    # gemmini = divide_dim(gemmini, gemmini.forward(a_alloc), 1, 64)
    # gemmini = divide_dim(gemmini, gemmini.forward(a_alloc), 0, 16)
    # # gemmini = divide_dim(gemmini, gemmini.forward(a_alloc), 3, 16)
    gemmini = simplify(gemmini)
    # print("after divide")
    # print(gemmini)
    # gemmini = rearrange_dim(gemmini, a_alloc, [0, 2, 1, 3])
    print(gemmini)

    gemmini = set_memory(gemmini, a_alloc, GEMM_SCRATCH)
    gemmini = set_memory(gemmini, b_alloc, GEMM_SCRATCH)
    gemmini = set_memory(gemmini, res_alloc, GEMM_ACCUM)
    # print(gemmini)
    tuples = [
        (ld_i8_block_id1_new, ld_i8_block_id1_v2_new),
        (ld_i8_block_id2_new, ld_i8_block_id2_v2_new),
        (matmul_acc_i8, matmul_acc_i8_v2),
        (st_acc_i8, st_acc_i8_v2),
    ]
    for t in tuples:
        gemmini = simplify(replace_and_inline(gemmini, t))
    print("after replace load buffers + GEMM + Store")
    print(gemmini)
    # print(gemmini.c_code_str())
    # problem: due to merging and tiling the pattern doesn't match!--> solved

    # replace GEMM + Store : added above


def test3():
    KK = 512
    M = 1024
    N = 256
    cpu = rename(
        matmul_algorithm(), "matmul_on_cpu"
    )  # Rename "matmul" to "matmul_on_cpu"
    cpu = cpu.partial_eval(N=N, M=M, K=KK)
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
    # Grab cursors
    i_loop = gemmini.find_loop("i")
    j_loop = gemmini.find_loop("j")
    k_loop = gemmini.find_loop("k")
    res_load = gemmini.find("res = D[_]")
    a_assign = gemmini.find("a2 = A[_]")
    b_assign = gemmini.find("b2 = B[_]")
    res_alloc = res_load.prev()

    # Handling Tilings
    gemmini, [ii, ji] = tile_loops(gemmini, [(i_loop, 16), (j_loop, 16)])

    gemmini, [ki] = tile_loops(gemmini, [(k_loop, 16)])
    gemmini = divide_loop(gemmini, j_loop, 4, ["joo", "joi"], perfect=True)

    # first reordering + replace_init_out
    gemmini = reorder_loops_from_idx_zz(
        gemmini, res_alloc, ["joo", "joi", "io", "ii", "ji"]
    )

    gemmini, _ = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
    gemmini = fission_as_much_as_possible(gemmini, res_load)
    gemmini = set_memory(gemmini, res_alloc, GEMM_ACCUM)
    gemmini = simplify(
        replace_and_inline(gemmini, (ld_acc_i32_vector, ld_acc_i32_vector_v2))
    )

    # print("after replace init out")
    # print(gemmini)
    # complete reordering
    gemmini = fission_as_much_as_possible(gemmini, k_loop)
    gemmini = reorder_loops_from_idx_zz(
        gemmini, a_assign, ["joo", "ko", "joi", "io", "ii", "ji", "ki"]
    )
    # print("after complete reordering")
    # print(gemmini)

    # replace load buffers
    # Step 1: lift allocation
    (gemmini, occ_size), a_load, a_alloc = bind_and_lift(
        gemmini, gemmini.forward(a_assign).rhs(), max_size=sc_size
    )
    (gemmini, _), b_load, b_alloc = bind_and_lift(
        gemmini, gemmini.forward(b_assign).rhs(), max_size=sc_size - occ_size
    )
    # Step 2: fission
    gemmini = fission_as_much_as_possible(gemmini, a_load)
    gemmini = fission_as_much_as_possible(gemmini, b_load)

    # Step 2.5: remove redundant loops
    gemmini = remove_redundant_loops(
        gemmini, a_load, num=2
    )  # original 2: check for final zizag mapping (fusion)
    gemmini = remove_redundant_loops(gemmini, b_load, num=2)
    print(gemmini)

    gemmini = rearrange_dim(gemmini, a_alloc, [1, 0, 2, 3])
    gemmini = rearrange_dim(gemmini, b_alloc, [0, 1, 3, 2])
    gemmini = reorder_loops_from_idx(gemmini, a_load)
    gemmini = reorder_loops_from_idx(gemmini, b_load)
    print(gemmini)
    # print(idx_list(gemmini.forward(a_load)))

    loop_list = get_loops_at_or_above(gemmini.forward(a_load))
    print([l.name() for l in loop_list])
    gemmini, _ = tile_loops(gemmini, [(loop_list[2], 4)])
    print(gemmini)
    # gemmini = divide_dim(gemmini, a_alloc, 0, 4)
    # gemmini = simplify(gemmini)
    # gemmini = rearrange_dim(gemmini, a_alloc, [0, 2, 3, 1, 4])
    # gemmini = reorder_loops_from_idx(gemmini, a_load)s
    # loop_list = get_loops_at_or_above(gemmini.forward(a_load))
    # gemmini = mult_loops(gemmini, loop_list[-2], 'k')
    # gemmini = simplify(gemmini)
    # gemmini = reorder_loops_from_idx_zz(gemmini, a_load, ['koo', 'io', 'ii', 'koi', 'ki'])

    # loop_list = get_loops_at_or_above(gemmini.forward(a_load))
    # gemmini, _ = tile_loops(gemmini, [(loop_list[-4], 4)])
    # gemmini = rearrange_dim(gemmini, a_alloc, [1, 2, 0, 3])
    # gemmini = reorder_loops_from_idx(gemmini, a_load)

    gemmini = set_memory(gemmini, a_alloc, GEMM_SCRATCH)
    gemmini = set_memory(gemmini, b_alloc, GEMM_SCRATCH)
    gemmini = set_memory(gemmini, res_alloc, GEMM_ACCUM)
    # print(gemmini)
    tuples = [
        (ld_i8_block_id1, ld_i8_block_id1_v2),
        (ld_i8_block_id2, ld_i8_block_id2_v2),
        (matmul_acc_i8, matmul_acc_i8_v2),
        (st_acc_i8, st_acc_i8_v2),
    ]
    for t in tuples:
        gemmini = simplify(replace_and_inline(gemmini, t))
    print("after replace load buffers + GEMM + Store")
    print(gemmini)
    print(gemmini.c_code_str())


def test4():
    KK = 512
    M = 1024
    N = 256
    cpu = rename(
        matmul_algorithm(), "matmul_on_cpu"
    )  # Rename "matmul" to "matmul_on_cpu"
    cpu = cpu.partial_eval(N=N, M=M, K=KK)
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
    # Grab cursors
    i_loop = gemmini.find_loop("i")
    j_loop = gemmini.find_loop("j")
    k_loop = gemmini.find_loop("k")
    res_load = gemmini.find("res = D[_]")
    a_assign = gemmini.find("a2 = A[_]")
    b_assign = gemmini.find("b2 = B[_]")
    res_alloc = res_load.prev()

    # Handling Tilings
    gemmini, [ii, ji] = tile_loops(gemmini, [(i_loop, 16), (j_loop, 16)])

    gemmini, [ki] = tile_loops(gemmini, [(k_loop, 16)])
    gemmini = divide_loop(gemmini, j_loop, 4, ["joo", "joi"], perfect=True)

    # first reordering + replace_init_out
    gemmini = reorder_loops_from_idx_zz(
        gemmini, res_alloc, ["joo", "joi", "io", "ii", "ji"]
    )

    gemmini, _ = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
    gemmini = fission_as_much_as_possible(gemmini, res_load)
    gemmini = set_memory(gemmini, res_alloc, GEMM_ACCUM)
    gemmini = simplify(
        replace_and_inline(gemmini, (ld_acc_i32_vector, ld_acc_i32_vector_v2))
    )

    # print("after replace init out")
    # print(gemmini)
    # complete reordering
    gemmini = fission_as_much_as_possible(gemmini, k_loop)
    gemmini = reorder_loops_from_idx_zz(
        gemmini, a_assign, ["joo", "ko", "joi", "io", "ii", "ji", "ki"]
    )
    # print("after complete reordering")
    # print(gemmini)

    # replace load buffers
    # Step 1: lift allocation
    (gemmini, occ_size), a_load, a_alloc = bind_and_lift(
        gemmini, gemmini.forward(a_assign).rhs(), max_size=sc_size
    )
    (gemmini, _), b_load, b_alloc = bind_and_lift(
        gemmini, gemmini.forward(b_assign).rhs(), max_size=sc_size - occ_size
    )
    # Step 2: fission
    gemmini = fission_as_much_as_possible(gemmini, a_load)
    gemmini = fission_as_much_as_possible(gemmini, b_load)

    # Step 2.5: remove redundant loops
    gemmini = remove_redundant_loops(
        gemmini, a_load, num=2
    )  # original 2: check for final zizag mapping (fusion)
    gemmini = remove_redundant_loops(gemmini, b_load, num=2)

    print(gemmini)
    # Step 3: restructure for pattern matching -> 1. back to the original dimension for each tensor + reordering/multiplying loops accordingly (4 steps),
    # 2. tiling to match the instruction patterns 3. Set the memories + call the replace
    gemmini = rearrange_dim(gemmini, a_alloc, [1, 2, 0, 3])
    gemmini = rearrange_dim(gemmini, b_alloc, [0, 3, 1, 2])
    gemmini = reorder_loops_from_idx(gemmini, a_load)
    gemmini = reorder_loops_from_idx(gemmini, b_load)
    # print(gemmini)

    # print(gemmini)
    # #idea: multiply the same loops then tile it using tile_loops to match the order and values and then try new intrinsics
    loop_list = get_loops_at_or_above(gemmini.forward(a_load))
    gemmini = mult_loops(gemmini, loop_list[-2], "k_a")
    gemmini = mult_loops(gemmini, loop_list[-4], "i_a")
    gemmini = simplify(gemmini)

    loop_list = get_loops_at_or_above(gemmini.forward(b_load))
    gemmini = mult_loops(gemmini, loop_list[-2], "j_b")
    gemmini = mult_loops(gemmini, loop_list[-4], "k_b")
    gemmini = simplify(gemmini)
    gemmini, _ = tile_loops(
        gemmini, [(gemmini.find_loop("i_a"), 16), (gemmini.find_loop("k_a"), 16)]
    )
    gemmini, _ = tile_loops(gemmini, [(gemmini.find_loop("k_ao"), 4)])
    gemmini, _ = tile_loops(
        gemmini, [(gemmini.find_loop("k_b"), 16), (gemmini.find_loop("j_b"), 16)]
    )
    gemmini, _ = tile_loops(gemmini, [(gemmini.find_loop("j_bo"), 4)])
    gemmini = rearrange_dim(gemmini, a_alloc, [0, 2, 1, 3])
    gemmini = rearrange_dim(gemmini, b_alloc, [0, 2, 1, 3])
    gemmini = simplify(gemmini)
    print(gemmini)

    gemmini = set_memory(gemmini, a_alloc, GEMM_SCRATCH)
    gemmini = set_memory(gemmini, b_alloc, GEMM_SCRATCH)
    gemmini = set_memory(gemmini, res_alloc, GEMM_ACCUM)
    # print(gemmini)
    tuples = [
        (ld_i8_block_id1, ld_i8_block_id1_v2),
        (ld_i8_block_id2, ld_i8_block_id2_v2),
        (matmul_acc_i8, matmul_acc_i8_v2),
        (st_acc_i8, st_acc_i8_v2),
    ]
    for t in tuples:
        gemmini = simplify(replace_and_inline(gemmini, t))
    print("after replace load buffers + GEMM + Store")
    print(gemmini)
    print(gemmini.c_code_str())


def test5_kij():
    KK = 1024
    M = 1024
    N = 1024
    cpu = rename(
        matmul_algorithm2(), "matmul_on_cpu"
    )  # Rename "matmul" to "matmul_on_cpu"
    cpu = cpu.partial_eval(N=N, M=M, K=KK)
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
    # Grab cursors
    i_loop = gemmini.find_loop("i")
    j_loop = gemmini.find_loop("j")
    k_loop = gemmini.find_loop("k")
    res_load = gemmini.find("res = D[_]")
    a_assign = gemmini.find("a2 = A[_]")
    b_assign = gemmini.find("b2 = B[_]")
    res_alloc = res_load.prev()

    # Handling Tilings
    # gemmini, [ii, ji] = tile_loops(gemmini, [(i_loop, 16), (j_loop, 16)])
    gemmini = reorder_loops(gemmini, "i j")
    print(gemmini)

    # gemmini, [ki] = tile_loops(gemmini, [(k_loop, 16)])
    # gemmini = divide_loop(gemmini, j_loop, 4, ["joo", "joi"], perfect=True)

    # #first reordering + replace_init_out
    # gemmini = reorder_loops_from_idx_zz(gemmini, res_alloc, ['joo', 'joi', 'io', 'ii', 'ji'])

    # gemmini, _ = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
    # gemmini = fission_as_much_as_possible(gemmini, res_load)
    # gemmini = set_memory(gemmini, res_alloc, GEMM_ACCUM)
    # gemmini = simplify(replace_and_inline(gemmini, (ld_acc_i32_vector, ld_acc_i32_vector_v2)))

    # # print("after replace init out")
    # # print(gemmini)
    # # complete reordering
    # gemmini = fission_as_much_as_possible(gemmini, k_loop)
    # gemmini = reorder_loops_from_idx_zz(gemmini, a_assign, ['joo', 'ko', 'joi', 'io', 'ii', 'ji', 'ki'])
    # # print("after complete reordering")
    # # print(gemmini)

    # #replace load buffers
    # # Step 1: lift allocation
    # (gemmini, occ_size),  a_load, a_alloc = bind_and_lift(
    #     gemmini, gemmini.forward(a_assign).rhs(), max_size=sc_size
    # )
    # (gemmini, _), b_load, b_alloc = bind_and_lift(
    #     gemmini, gemmini.forward(b_assign).rhs(), max_size=sc_size - occ_size
    # )
    # # Step 2: fission
    # gemmini = fission_as_much_as_possible(gemmini, a_load)
    # gemmini = fission_as_much_as_possible(gemmini, b_load)

    # # Step 2.5: remove redundant loops
    # gemmini = remove_redundant_loops(gemmini, a_load, num=2) # original 2: check for final zizag mapping (fusion)
    # gemmini = remove_redundant_loops(gemmini, b_load, num=2)

    # print(gemmini)
    # # Step 3: restructure for pattern matching -> 1. back to the original dimension for each tensor + reordering/multiplying loops accordingly (4 steps),
    # # 2. tiling to match the instruction patterns 3. Set the memories + call the replace
    # gemmini = rearrange_dim(gemmini, a_alloc, [1, 2, 0, 3])
    # gemmini = rearrange_dim(gemmini, b_alloc, [0, 3, 1, 2])
    # gemmini = reorder_loops_from_idx(gemmini, a_load)
    # gemmini = reorder_loops_from_idx(gemmini, b_load)
    # # print(gemmini)

    # # print(gemmini)
    # # #idea: multiply the same loops then tile it using tile_loops to match the order and values and then try new intrinsics
    # loop_list = get_loops_at_or_above(gemmini.forward(a_load))
    # gemmini = mult_loops(gemmini, loop_list[-2], 'k_a')
    # gemmini = mult_loops(gemmini, loop_list[-4], 'i_a')
    # gemmini = simplify(gemmini)

    # loop_list = get_loops_at_or_above(gemmini.forward(b_load))
    # gemmini = mult_loops(gemmini, loop_list[-2], 'j_b')
    # gemmini = mult_loops(gemmini, loop_list[-4], 'k_b')
    # gemmini = simplify(gemmini)
    # gemmini, _ = tile_loops(gemmini, [(gemmini.find_loop('i_a'), 16), (gemmini.find_loop('k_a'), 16)])
    # gemmini, _ = tile_loops(gemmini, [(gemmini.find_loop('k_ao'), 4)])
    # gemmini, _ = tile_loops(gemmini, [(gemmini.find_loop('k_b'), 16), (gemmini.find_loop('j_b'), 16 )])
    # gemmini, _ = tile_loops(gemmini, [(gemmini.find_loop('j_bo'), 4 )])
    # gemmini = rearrange_dim(gemmini, a_alloc, [0, 2, 1, 3])
    # gemmini = rearrange_dim(gemmini, b_alloc, [0, 2, 1, 3])
    # gemmini = simplify(gemmini)
    # print(gemmini)

    # gemmini = set_memory(gemmini, a_alloc, GEMM_SCRATCH)
    # gemmini = set_memory(gemmini, b_alloc, GEMM_SCRATCH)
    # gemmini = set_memory(gemmini, res_alloc, GEMM_ACCUM)
    # # print(gemmini)
    # tuples = [
    #     (ld_i8_block_id1, ld_i8_block_id1_v2),
    #     (ld_i8_block_id2, ld_i8_block_id2_v2),
    #     (matmul_acc_i8, matmul_acc_i8_v2),
    #     (st_acc_i8, st_acc_i8_v2),
    # ]
    # for t in tuples:
    #     gemmini = simplify(replace_and_inline(gemmini, t))
    # print("after replace load buffers + GEMM + Store")
    # print(gemmini)
    # print(gemmini.c_code_str())


def get_loops_at_or_above(cursor):
    loops = []
    while isinstance((parent := cursor.parent()), pc.ForCursor):
        loops.append(parent)
        cursor = parent
    return list(reversed(loops))


# test_matmul_zz_sample3_2()
# test_matmul_zz_2()
# test2()
# test3()
# test4()
# test5_kij()


def test_instr():
    proc = ld_i8_block_id1
    print(proc.body)
    print(proc._loopir_proc)
    print(proc._loopir_proc.body)


test_instr()
