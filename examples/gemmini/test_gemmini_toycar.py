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


def test_matmul_toycar(NN, MM, KK):

    assert NN == 1, "N in toycar layers is one!"

    cpu = rename(
        matmul_algorithm(), "matmul_on_cpu"
    )  # Rename "matmul" to "matmul_on_cpu"
    cpu = cpu.partial_eval(K=KK, N=NN, M=MM)

    # cpu = cpu.add_assertion("N % 256 == 0")
    # cpu = cpu.add_assertion("M % 256 == 0")

    # Rename the procedure to "matmul_on_gemmini"
    gemmini = rename(cpu, "toycar_layer_{}_{}_{}".format(NN, MM, KK))

    # print("")
    # print("===== THIS IS THE ORIGINAL MATMUL ALGORITHM BEFORE SCHEDULING ====")
    # print(gemmini)
    # print("===== THIS IS THE ORIGINAL MATMUL ALGORITHM BEFORE SCHEDULING ====")
    # print("")

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

    # aliases
    j_loop_factor = j_loop._impl._node.hi.val
    k_loop_factor = k_loop._impl._node.hi.val
    # Schedule starts here!!

    # Tile loops for a scratchpad and an accumulator
    print(j_loop._impl._node.hi, j_loop._impl._node.lo)
    print(type(i_loop._impl._node.hi))
    # print(dir(j_loop._impl))

    if MM >= 16:
        gemmini, _ = tile_loops(gemmini, [(i_loop, 1), (j_loop, 16)], perfect=True)
        # print(i_loop)
        if MM % 512 == 0:
            gemmini, [j_outer] = tile_loops(gemmini, [(j_loop, 16)])
            # print(gemmini)
            # gemmini = reorder_loops(gemmini, gemmini.find_loop(gemmini.forward(j_loop).name()))
    else:
        gemmini, _ = tile_loops(
            gemmini, [(i_loop, 1), (j_loop, j_loop_factor)], perfect=True
        )

    if KK >= 16:
        gemmini, _ = tile_loops(gemmini, [(k_loop, 16)])
    else:
        gemmini, _ = tile_loops(gemmini, [(k_loop, k_loop_factor)])
    print("gemmini after first tiling")
    print(gemmini)

    # # Bind and lift scratchpad & accumulator memories
    gemmini = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
    gemmini, a_load, a_alloc = bind_and_lift(
        gemmini, gemmini.forward(a_assign).rhs(), max_size=sc_size / 2
    )
    gemmini, b_load, b_alloc = bind_and_lift(
        gemmini, gemmini.forward(b_assign).rhs(), max_size=sc_size / 2
    )
    print("gemmini after memory allocation")
    print(gemmini)

    # update the factors
    k_loop_factor = gemmini.forward(k_loop)._impl._node.hi.val
    # # Divide by 4 to use load_blocks
    if KK >= 64:
        gemmini, _ = tile_loops(gemmini, [(k_loop, 4)])
    else:
        gemmini, _ = tile_loops(gemmini, [(k_loop, k_loop_factor)])
    if MM >= 256:
        if MM % 512 == 0:
            gemmini, [j_imost] = tile_loops(gemmini, [(j_outer, 4)], perfect=True)
        else:
            gemmini, [j_imost] = tile_loops(gemmini, [(j_loop, 4)], perfect=True)
    elif MM >= 64:
        gemmini, [j_imost] = tile_loops(gemmini, [(j_loop, 4)], perfect=True)

    else:
        j_imost = gemmini.forward(j_loop)
    print("gemmini after second tiling")
    print(gemmini)
    # # Fission all the loops
    gemmini = fission_as_much_as_possible(gemmini, res_load)
    # print("5: After fissioning res_load")
    # print(gemmini)
    gemmini = fission_as_much_as_possible(gemmini, k_loop)
    # print("6: After fissioning k_loop")
    # print(gemmini)
    gemmini = fission_as_much_as_possible(gemmini, a_load)
    # print("7: After fissioning a_load")
    # print(gemmini)
    gemmini = fission_as_much_as_possible(gemmini, b_load)
    print("8: After fissioning")
    print(gemmini)

    # # Fix indexing
    # if MM > 16:
    #     gemmini = rearrange_dim(gemmini, a_alloc, [0, 2, 1, 3])
    #     gemmini = rearrange_dim(gemmini, b_alloc, [2, 0, 3, 1])
    # else:
    #     # gemmini = rearrange_dim(gemmini, a_alloc, [0, 1, 2, 3])
    #     gemmini = rearrange_dim(gemmini, b_alloc, [1, 0, 2])

    gemmini = rearrange_dim(gemmini, a_alloc, [0, 2, 1, 3])
    gemmini = rearrange_dim(gemmini, b_alloc, [2, 0, 3, 1])
    gemmini = reorder_loops_from_idx(gemmini, a_load)
    gemmini = reorder_loops_from_idx(gemmini, b_load)
    gemmini = reorder_loops_from_idx(
        gemmini, gemmini.forward(a_assign).rhs()
    )  # you don't need to do this, it seems to be done automatically
    gemmini = remove_redundant_loops(gemmini, a_load, num=2)
    gemmini = remove_redundant_loops(gemmini, b_load, num=1)
    print("9: After rearranging dimensions")
    print(gemmini)

    # # Replace to gemmini calls, inline to v2, and hoist all the configurations
    gemmini = set_memory(gemmini, res_alloc, GEMM_ACCUM)
    gemmini = set_memory(gemmini, a_alloc, GEMM_SCRATCH)
    gemmini = set_memory(gemmini, b_alloc, GEMM_SCRATCH)
    # print("10: After setting memories")
    # print(gemmini)
    if KK == 8:
        ld_i8_block_id1_intr = ld_i8_block_id1_new
        ld_i8_block_id1_v2_intr = ld_i8_block_id1_v2_new
        do_id1_name = "do_ld_i8_block_id1_new(_)"
    else:
        ld_i8_block_id1_intr = ld_i8_block_id1
        ld_i8_block_id1_v2_intr = ld_i8_block_id1_v2
        do_id1_name = "do_ld_i8_block_id1(_)"
    print(ld_i8_block_id1_intr)
    tuples = [
        (ld_acc_i32_vector, ld_acc_i32_vector_v2),
        (ld_i8_block_id1_intr, ld_i8_block_id1_v2_intr),
        (ld_i8_block_id2, ld_i8_block_id2_v2),
        (matmul_acc_i8, matmul_acc_i8_v2),
        (st_acc_i8, st_acc_i8_v2),
    ]
    for t in tuples:
        gemmini = simplify(replace_and_inline(gemmini, t))
    print("11: After replacing and inlining")
    print(gemmini)
    # # Add a guard to redundant loads
    gemmini = add_guard(gemmini, gemmini.find(do_id1_name))
    gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id2(_)"))
    # print("12: After adding guards")
    # print(gemmini)

    # # fuse loops, and unroll
    gemmini = fuse_all_loops(gemmini, gemmini.body()[0])
    # print("13: After fusing all loops")
    # print(gemmini)
    gemmini = unroll_all(gemmini, gemmini.find_loop(j_imost.name(), many=True))
    # print("14: After unrolling j_imost")
    # print(gemmini)
    # # Schedule ends here!!
    # # 29 lines excluding comments and newlines

    gemmini = simplify(gemmini)  # mathematical simplifications 2*16 -> 32, etc.

    print("")
    print("============= THIS IS THE SCHEDULED MATMUL ===============")
    print(gemmini)
    print("============= THIS IS THE SCHEDULED MATMUL ===============")
    print("")
    print(gemmini.c_code_str())
    return cpu, gemmini


def test_matmul_toycar_1_8_128():
    NN = 1
    MM = 8
    KK = 128
    # assert NN == 1, "N in toycar layers is one!"

    cpu = rename(
        matmul_algorithm(), "matmul_on_cpu"
    )  # Rename "matmul" to "matmul_on_cpu"
    cpu = cpu.partial_eval(K=KK, N=NN, M=MM)

    # cpu = cpu.add_assertion("N % 256 == 0")
    # cpu = cpu.add_assertion("M % 256 == 0")

    # Rename the procedure to "matmul_on_gemmini"
    gemmini = rename(cpu, "toycar_layer_{}_{}_{}".format(NN, MM, KK))

    # print("")
    # print("===== THIS IS THE ORIGINAL MATMUL ALGORITHM BEFORE SCHEDULING ====")
    # print(gemmini)
    # print("===== THIS IS THE ORIGINAL MATMUL ALGORITHM BEFORE SCHEDULING ====")
    # print("")

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

    # aliases
    j_loop_factor = j_loop._impl._node.hi.val

    # Schedule starts here!!

    # Tile loops for a scratchpad and an accumulator
    print(j_loop._impl._node.hi, j_loop._impl._node.lo)
    print(type(i_loop._impl._node.hi))
    # print(dir(j_loop._impl))

    gemmini, _ = tile_loops(
        gemmini, [(i_loop, 1), (j_loop, j_loop_factor)], perfect=True
    )

    if KK >= 16:
        gemmini, _ = tile_loops(gemmini, [(k_loop, 16)])

    print("gemmini after first tiling")
    print(gemmini)

    # # Bind and lift scratchpad & accumulator memories
    gemmini = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
    gemmini, a_load, a_alloc = bind_and_lift(
        gemmini, gemmini.forward(a_assign).rhs(), max_size=sc_size / 2
    )
    gemmini, b_load, b_alloc = bind_and_lift(
        gemmini, gemmini.forward(b_assign).rhs(), max_size=sc_size / 2
    )
    print("gemmini after memory allocation")
    print(gemmini)
    # # Divide by 4 to use load_blocks
    if KK >= 64:
        gemmini, _ = tile_loops(gemmini, [(k_loop, 4)])
    else:
        gemmini, _ = tile_loops(gemmini, [(k_loop, 1)])

    j_imost = gemmini.forward(j_loop)
    print("gemmini after second tiling")
    print(gemmini)
    # # Fission all the loops
    gemmini = fission_as_much_as_possible(gemmini, res_load)
    # print("5: After fissioning res_load")
    # print(gemmini)
    gemmini = fission_as_much_as_possible(gemmini, k_loop)
    # print("6: After fissioning k_loop")
    # print(gemmini)
    gemmini = fission_as_much_as_possible(gemmini, a_load)
    # print("7: After fissioning a_load")
    # print(gemmini)
    gemmini = fission_as_much_as_possible(gemmini, b_load)
    print("8: After fissioning")
    print(gemmini)

    # # Fix indexing
    # if MM > 16:
    #     gemmini = rearrange_dim(gemmini, a_alloc, [0, 2, 1, 3])
    #     gemmini = rearrange_dim(gemmini, b_alloc, [2, 0, 3, 1])
    # else:
    #     # gemmini = rearrange_dim(gemmini, a_alloc, [0, 1, 2, 3])
    #     gemmini = rearrange_dim(gemmini, b_alloc, [1, 0, 2])

    gemmini = rearrange_dim(gemmini, a_alloc, [0, 2, 1, 3])
    gemmini = rearrange_dim(gemmini, b_alloc, [2, 0, 3, 1])
    gemmini = reorder_loops_from_idx(gemmini, a_load)
    gemmini = reorder_loops_from_idx(gemmini, b_load)
    gemmini = reorder_loops_from_idx(
        gemmini, gemmini.forward(a_assign).rhs()
    )  # you don't need to do this, it seems to be done automatically
    gemmini = remove_redundant_loops(gemmini, a_load, num=2)
    gemmini = remove_redundant_loops(gemmini, b_load, num=1)
    print("9: After rearranging dimensions")
    print(gemmini)

    # # Replace to gemmini calls, inline to v2, and hoist all the configurations
    gemmini = set_memory(gemmini, res_alloc, GEMM_ACCUM)
    gemmini = set_memory(gemmini, a_alloc, GEMM_SCRATCH)
    gemmini = set_memory(gemmini, b_alloc, GEMM_SCRATCH)
    # print("10: After setting memories")
    # print(gemmini)
    print(ld_i8_block_id2_new)
    tuples = [
        (ld_acc_i32_vector, ld_acc_i32_vector_v2),
        (ld_i8_block_id1, ld_i8_block_id1_v2),
        (ld_i8_block_id2_new, ld_i8_block_id2_v2_new),
        (matmul_acc_i8, matmul_acc_i8_v2),
        (st_acc_i8, st_acc_i8_v2),
    ]
    for t in tuples:
        gemmini = simplify(replace_and_inline(gemmini, t))
    print("11: After replacing and inlining")
    print(gemmini)
    # # Add a guard to redundant loads
    gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id1(_)"))
    gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id2_new(_)"))
    # print("12: After adding guards")
    # print(gemmini)

    # # fuse loops, and unroll
    gemmini = fuse_all_loops(gemmini, gemmini.body()[0])
    # print("13: After fusing all loops")
    # print(gemmini)
    gemmini = unroll_all(gemmini, gemmini.find_loop(j_imost.name(), many=True))
    # print("14: After unrolling j_imost")
    # print(gemmini)
    # # Schedule ends here!!
    # # 29 lines excluding comments and newlines

    gemmini = simplify(gemmini)  # mathematical simplifications 2*16 -> 32, etc.

    print("")
    print("============= THIS IS THE SCHEDULED MATMUL ===============")
    print(gemmini)
    print("============= THIS IS THE SCHEDULED MATMUL ===============")
    print("")
    print(gemmini.c_code_str())
    return cpu, gemmini


# --- Toycar tests ---
# test_matmul_toycar(1, 128, 640)
# test_matmul_toycar(1, 128, 128)
# test_matmul_toycar_1_8_128()
# test_matmul_toycar(1, 128, 8)
test_matmul_toycar(1, 640, 128)
