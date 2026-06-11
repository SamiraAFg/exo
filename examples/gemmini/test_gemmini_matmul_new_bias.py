from __future__ import annotations

from exo.platforms.gemmini import (
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


def tile_and_autolift_alloc(proc, loops, alloc_c, dep_set=None, max_size=0, lift=True):
    proc, _ = tile_loops(proc, loops, perfect=True)
    alloc_c = p.forward(alloc_c)
    loop_c = get_enclosing_loop(p, alloc_c)
    accum_size = 1
    while True:
        try:
            if not isinstance(loop_c, pc.ForCursor):
                break
            if dep_set == None or loop_c.name() in dep_set:
                if (
                    isinstance(loop_c.hi(), LiteralCursor)
                    and accum_size * loop_c.hi().value() <= max_size
                ):

                    p = expand_dim(p, alloc_c, loop_c.hi().value(), loop_c.name())
                    accum_size = accum_size * loop_c.hi().value()
            if lift:
                p = lift_alloc(p, alloc_c)
            loop_c = loop_c.parent()
        except:
            break
    return p


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


def test_matmul_256():
    KK = 256
    NN = 256
    MM = 256

    cpu = rename(
        matmul_algorithm(), "cpu_matmul_256"
    )  # Rename "matmul" to "matmul_on_cpu"
    # cpu = cpu.partial_eval(K=KK)
    cpu = cpu.partial_eval(N=NN, M=MM, K=KK)
    # cpu = cpu.add_assertion("N % 256 == 0")
    # cpu = cpu.add_assertion("M % 256 == 0")

    # Rename the procedure to "matmul_on_gemmini"
    gemmini = rename(cpu, "matmul_256")

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

    # Schedule starts here!!

    # Tile loops for a scratchpad and an accumulator
    gemmini, _ = tile_loops(gemmini, [(i_loop, 16), (j_loop, 16)], perfect=True)
    # gemmini, [_, j_outer] = tile_loops(
    #     gemmini, [(i_loop, 16), (j_loop, 16)], perfect=True
    # )
    gemmini, _ = tile_loops(gemmini, [(k_loop, 16)])
    # gemmini, _ = tile_loops(gemmini, [(i_loop, 4)])

    # Bind and lift scratchpad & accumulator memories
    gemmini = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
    gemmini, a_load, a_alloc = bind_and_lift(
        gemmini, gemmini.forward(a_assign).rhs(), max_size=sc_size
    )
    gemmini, b_load, b_alloc = bind_and_lift(
        gemmini, gemmini.forward(b_assign).rhs(), max_size=sc_size
    )

    # Divide by 4 to use load_blocks
    gemmini, _ = tile_loops(gemmini, [(k_loop, 4)])
    gemmini, [j_imost] = tile_loops(gemmini, [(j_loop, 4)], perfect=True)

    # Fission all the loops
    gemmini = fission_as_much_as_possible(gemmini, res_load)
    gemmini = fission_as_much_as_possible(gemmini, k_loop)
    gemmini = fission_as_much_as_possible(gemmini, a_load)
    gemmini = fission_as_much_as_possible(gemmini, b_load)
    # print("5: After fissioning res_load")
    # print(gemmini)

    # Fix indexing
    gemmini = rearrange_dim(gemmini, a_alloc, [0, 2, 1, 3])
    gemmini = rearrange_dim(gemmini, b_alloc, [2, 0, 3, 1])
    gemmini = reorder_loops_from_idx(gemmini, a_load)
    gemmini = reorder_loops_from_idx(gemmini, b_load)
    gemmini = reorder_loops_from_idx(gemmini, gemmini.forward(a_assign).rhs())
    # print(gemmini)
    gemmini = remove_redundant_loops(gemmini, a_load, num=3)
    gemmini = remove_redundant_loops(gemmini, b_load, num=2)

    # # # Replace to gemmini calls, inline to v2, and hoist all the configurations
    gemmini = set_memory(gemmini, res_alloc, GEMM_ACCUM)
    gemmini = set_memory(gemmini, a_alloc, GEMM_SCRATCH)
    gemmini = set_memory(gemmini, b_alloc, GEMM_SCRATCH)
    tuples = [
        # (zero_acc_i32, zero_acc_i32_v2),
        (ld_acc_i32_vector, ld_acc_i32_vector_v2),
        (ld_i8_block_id1, ld_i8_block_id1_v2),
        (ld_i8_block_id2, ld_i8_block_id2_v2),
        (matmul_acc_i8, matmul_acc_i8_v2),
        (st_acc_i8, st_acc_i8_v2),
    ]
    for t in tuples:
        gemmini = simplify(replace_and_inline(gemmini, t))

    # # Add a guard to redundant loads
    gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id1(_)"))
    gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id2(_)"))

    # # fuse loops, and unroll
    gemmini = fuse_all_loops(gemmini, gemmini.body()[0])
    gemmini = unroll_all(gemmini, gemmini.find_loop(j_imost.name(), many=True))
    # print(gemmini)

    # # Schedule ends here!!
    # # 29 lines excluding comments and newlines

    gemmini = simplify(gemmini)

    print("")
    print("============= THIS IS THE SCHEDULED MATMUL ===============")
    print(gemmini)
    print("============= THIS IS THE SCHEDULED MATMUL ===============")
    print("")
    return cpu, gemmini


def test_matmul_128():
    KK = 128
    NN = 128
    MM = 128

    cpu = rename(
        matmul_algorithm(), "cpu_matmul_128"
    )  # Rename "matmul" to "matmul_on_cpu"
    # cpu = cpu.partial_eval(K=KK)
    cpu = cpu.partial_eval(N=NN, M=MM, K=KK)
    # cpu = cpu.add_assertion("N % 256 == 0")
    # cpu = cpu.add_assertion("M % 256 == 0")

    # Rename the procedure to "matmul_on_gemmini"
    gemmini = rename(cpu, "matmul_128")

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

    # Schedule starts here!!

    # Tile loops for a scratchpad and an accumulator
    gemmini, _ = tile_loops(gemmini, [(i_loop, 16), (j_loop, 16)], perfect=True)
    # gemmini, [_, j_outer] = tile_loops(
    #     gemmini, [(i_loop, 16), (j_loop, 16)], perfect=True
    # )
    gemmini, _ = tile_loops(gemmini, [(k_loop, 16)])

    # Bind and lift scratchpad & accumulator memories
    gemmini = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
    gemmini, a_load, a_alloc = bind_and_lift(
        gemmini, gemmini.forward(a_assign).rhs(), max_size=sc_size
    )
    gemmini, b_load, b_alloc = bind_and_lift(
        gemmini, gemmini.forward(b_assign).rhs(), max_size=sc_size
    )

    # Divide by 4 to use load_blocks
    gemmini, _ = tile_loops(gemmini, [(k_loop, 4)])
    gemmini, [j_imost] = tile_loops(gemmini, [(j_loop, 4)], perfect=True)

    # Fission all the loops
    gemmini = fission_as_much_as_possible(gemmini, res_load)
    gemmini = fission_as_much_as_possible(gemmini, k_loop)
    gemmini = fission_as_much_as_possible(gemmini, a_load)
    gemmini = fission_as_much_as_possible(gemmini, b_load)

    print("5: After fissioning res_load")

    # Fix indexing
    gemmini = rearrange_dim(gemmini, a_alloc, [0, 2, 1, 3])
    gemmini = rearrange_dim(gemmini, b_alloc, [2, 0, 3, 1])
    gemmini = reorder_loops_from_idx(gemmini, a_load)
    gemmini = reorder_loops_from_idx(gemmini, b_load)
    gemmini = reorder_loops_from_idx(gemmini, gemmini.forward(a_assign).rhs())
    gemmini = remove_redundant_loops(gemmini, a_load, num=3)
    gemmini = remove_redundant_loops(gemmini, b_load, num=2)

    # # Replace to gemmini calls, inline to v2, and hoist all the configurations
    gemmini = set_memory(gemmini, res_alloc, GEMM_ACCUM)
    gemmini = set_memory(gemmini, a_alloc, GEMM_SCRATCH)
    gemmini = set_memory(gemmini, b_alloc, GEMM_SCRATCH)
    tuples = [
        # (zero_acc_i32, zero_acc_i32_v2),
        (ld_acc_i32_vector, ld_acc_i32_vector_v2),
        (ld_i8_block_id1, ld_i8_block_id1_v2),
        (ld_i8_block_id2, ld_i8_block_id2_v2),
        (matmul_acc_i8, matmul_acc_i8_v2),
        (st_acc_i8, st_acc_i8_v2),
    ]
    for t in tuples:
        gemmini = simplify(replace_and_inline(gemmini, t))

    # # Add a guard to redundant loads
    gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id1(_)"))
    gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id2(_)"))

    # # fuse loops, and unroll
    gemmini = fuse_all_loops(gemmini, gemmini.body()[0])
    gemmini = unroll_all(gemmini, gemmini.find_loop(j_imost.name(), many=True))
    # print(gemmini)

    # # Schedule ends here!!
    # # 29 lines excluding comments and newlines

    gemmini = simplify(gemmini)

    print("")
    print("============= THIS IS THE SCHEDULED MATMUL ===============")
    print(gemmini)
    print("============= THIS IS THE SCHEDULED MATMUL ===============")
    print("")
    return cpu, gemmini


def test_matmul_64():
    KK = 64
    NN = 64
    MM = 64

    cpu = rename(
        matmul_algorithm(), "cpu_matmul_64"
    )  # Rename "matmul" to "matmul_on_cpu"
    # cpu = cpu.partial_eval(K=KK)
    cpu = cpu.partial_eval(N=NN, M=MM, K=KK)
    # cpu = cpu.add_assertion("N % 256 == 0")
    # cpu = cpu.add_assertion("M % 256 == 0")

    # Rename the procedure to "matmul_on_gemmini"
    gemmini = rename(cpu, "matmul_64")

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

    # Schedule starts here!!

    # Tile loops for a scratchpad and an accumulator
    gemmini, _ = tile_loops(gemmini, [(i_loop, 16), (j_loop, 16)], perfect=True)
    # gemmini, [_, j_outer] = tile_loops(
    #     gemmini, [(i_loop, 16), (j_loop, 16)], perfect=True
    # )
    gemmini, _ = tile_loops(gemmini, [(k_loop, 16)])

    # Bind and lift scratchpad & accumulator memories
    gemmini = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
    gemmini, a_load, a_alloc = bind_and_lift(
        gemmini, gemmini.forward(a_assign).rhs(), max_size=sc_size
    )
    gemmini, b_load, b_alloc = bind_and_lift(
        gemmini, gemmini.forward(b_assign).rhs(), max_size=sc_size
    )

    # Divide by 4 to use load_blocks
    gemmini, _ = tile_loops(gemmini, [(k_loop, 4)])
    gemmini, [j_imost] = tile_loops(gemmini, [(j_loop, 4)], perfect=True)

    # Fission all the loops
    gemmini = fission_as_much_as_possible(gemmini, res_load)
    gemmini = fission_as_much_as_possible(gemmini, k_loop)
    gemmini = fission_as_much_as_possible(gemmini, a_load)
    gemmini = fission_as_much_as_possible(gemmini, b_load)
    # print("5: After fissioning res_load")
    # print(gemmini)

    # Fix indexing
    gemmini = rearrange_dim(gemmini, a_alloc, [0, 2, 1, 3])
    gemmini = rearrange_dim(gemmini, b_alloc, [2, 0, 3, 1])
    gemmini = reorder_loops_from_idx(gemmini, a_load)
    gemmini = reorder_loops_from_idx(gemmini, b_load)
    gemmini = reorder_loops_from_idx(gemmini, gemmini.forward(a_assign).rhs())
    # print(gemmini)
    gemmini = remove_redundant_loops(gemmini, a_load, num=3)
    gemmini = remove_redundant_loops(gemmini, b_load, num=2)

    # # # Replace to gemmini calls, inline to v2, and hoist all the configurations
    gemmini = set_memory(gemmini, res_alloc, GEMM_ACCUM)
    gemmini = set_memory(gemmini, a_alloc, GEMM_SCRATCH)
    gemmini = set_memory(gemmini, b_alloc, GEMM_SCRATCH)
    tuples = [
        # (zero_acc_i32, zero_acc_i32_v2),
        (ld_acc_i32_vector, ld_acc_i32_vector_v2),
        (ld_i8_block_id1, ld_i8_block_id1_v2),
        (ld_i8_block_id2, ld_i8_block_id2_v2),
        (matmul_acc_i8, matmul_acc_i8_v2),
        (st_acc_i8, st_acc_i8_v2),
    ]
    for t in tuples:
        gemmini = simplify(replace_and_inline(gemmini, t))

    # # Add a guard to redundant loads
    gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id1(_)"))
    gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id2(_)"))

    # # fuse loops, and unroll
    gemmini = fuse_all_loops(gemmini, gemmini.body()[0])
    gemmini = unroll_all(gemmini, gemmini.find_loop(j_imost.name(), many=True))
    # print(gemmini)

    # # Schedule ends here!!
    # # 29 lines excluding comments and newlines

    gemmini = simplify(gemmini)

    print("")
    print("============= THIS IS THE SCHEDULED MATMUL ===============")
    print(gemmini)
    print("============= THIS IS THE SCHEDULED MATMUL ===============")
    print("")
    return cpu, gemmini


def test_matmul():
    KK = 512

    cpu = rename(
        matmul_algorithm(), "matmul_on_cpu"
    )  # Rename "matmul" to "matmul_on_cpu"
    cpu = cpu.partial_eval(K=KK)
    cpu = cpu.add_assertion("N % 256 == 0")
    cpu = cpu.add_assertion("M % 256 == 0")

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
    gemmini, _ = tile_loops(gemmini, [(i_loop, 16), (j_loop, 16)], perfect=True)
    # print(i_loop)
    gemmini, [_, j_outer] = tile_loops(
        gemmini, [(i_loop, 16), (j_loop, 16)], perfect=True
    )
    # print("j_outer points to:")
    # print(j_outer)
    gemmini, _ = tile_loops(gemmini, [(k_loop, 16)])
    # print(gemmini.forward(res_alloc))
    # # Bind and lift scratchpad & accumulator memories
    gemmini = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
    gemmini, a_load, a_alloc = bind_and_lift(
        gemmini, gemmini.forward(a_assign).rhs(), max_size=sc_size
    )
    gemmini, b_load, b_alloc = bind_and_lift(
        gemmini, gemmini.forward(b_assign).rhs(), max_size=sc_size
    )
    # print("gemmini before third tiling")
    # print(gemmini)
    # # Divide by 4 to use load_blocks
    gemmini, _ = tile_loops(gemmini, [(k_loop, 4)])
    gemmini, [j_imost] = tile_loops(gemmini, [(j_outer, 4)], perfect=True)
    print("before fissioning")
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
    # print("8: After fissioning")
    # print(gemmini)

    # # Fix indexing
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
    print("10: After setting memories")
    print(gemmini)

    tuples = [
        # (zero_acc_i32, zero_acc_i32_v2),
        (ld_acc_i32_vector, ld_acc_i32_vector_v2),
        (ld_i8_block_id1, ld_i8_block_id1_v2),
        (ld_i8_block_id2, ld_i8_block_id2_v2),
        (matmul_acc_i8, matmul_acc_i8_v2),
        (st_acc_i8, st_acc_i8_v2),
    ]
    for t in tuples:
        gemmini = simplify(replace_and_inline(gemmini, t))
    # print("11: After replacing and inlining")
    # print(gemmini)
    # # Add a guard to redundant loads
    gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id1(_)"))
    gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id2(_)"))
    # print("12: After adding guards")
    # print(gemmini)

    # # fuse loops, and unroll
    gemmini = fuse_all_loops(gemmini, gemmini.body()[0])
    # print("13: After fusing all loops")
    # print(gemmini)
    gemmini = unroll_all(gemmini, gemmini.find_loop(j_imost.name(), many=True))
    print("14: After unrolling j_imost")
    print(gemmini)
    # # Schedule ends here!!
    # # 29 lines excluding comments and newlines

    gemmini = simplify(gemmini)  # mathematical simplifications 2*16 -> 32, etc.

    print("")
    print("============= THIS IS THE SCHEDULED MATMUL ===============")
    print(gemmini)
    print("============= THIS IS THE SCHEDULED MATMUL ===============")
    print("")
    print(gemmini.c_code_str())
    # assert gemmini.c_code_str() == golden


def test_matmul_all(NN, MM, KK):
    # KK = 512
    assert NN % 64 == 0, "N must be a multiple of 64"
    assert MM % 64 == 0, "M must be a multiple of 64"
    assert KK % 64 == 0, "K must be a multiple of 64"
    cpu = rename(
        matmul_algorithm(), "matmul_on_cpu"
    )  # Rename "matmul" to "matmul_on_cpu"
    cpu = cpu.partial_eval(K=KK, N=NN, M=MM)

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
    gemmini, _ = tile_loops(gemmini, [(i_loop, 16), (j_loop, 16)], perfect=True)
    # print(i_loop)
    if NN % 512 == 0 and MM % 512 == 0:
        gemmini, [_, j_outer] = tile_loops(
            gemmini, [(i_loop, 16), (j_loop, 16)], perfect=True
        )

    gemmini, _ = tile_loops(gemmini, [(k_loop, 16)])
    # print(gemmini.forward(res_alloc))
    # # Bind and lift scratchpad & accumulator memories
    print("gemmini before autolift")
    print(gemmini)

    gemmini = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
    gemmini, a_load, a_alloc = bind_and_lift(
        gemmini, gemmini.forward(a_assign).rhs(), max_size=sc_size / 2
    )
    gemmini, b_load, b_alloc = bind_and_lift(
        gemmini, gemmini.forward(b_assign).rhs(), max_size=sc_size / 2
    )
    print("after autolift")
    print(gemmini)
    # # Divide by 4 to use load_blocks
    gemmini, _ = tile_loops(gemmini, [(k_loop, 4)])
    if NN % 512 == 0 and MM % 512 == 0:
        gemmini, [j_imost] = tile_loops(gemmini, [(j_outer, 4)], perfect=True)
    else:
        gemmini, [j_imost] = tile_loops(gemmini, [(j_loop, 4)], perfect=True)

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
    # print("8: After fissioning")
    # print(gemmini)

    # # Fix indexing
    gemmini = rearrange_dim(gemmini, a_alloc, [0, 2, 1, 3])
    gemmini = rearrange_dim(gemmini, b_alloc, [2, 0, 3, 1])
    gemmini = reorder_loops_from_idx(gemmini, a_load)
    gemmini = reorder_loops_from_idx(gemmini, b_load)
    gemmini = reorder_loops_from_idx(
        gemmini, gemmini.forward(a_assign).rhs()
    )  # you don't need to do this, it seems to be done automatically
    # print("gemmini after rearranging dimensions")
    # print(gemmini)
    gemmini = remove_redundant_loops(gemmini, a_load, num=2)
    gemmini = remove_redundant_loops(gemmini, b_load, num=1)
    # print("9: After rearranging dimensions")
    # print(gemmini)

    # # Replace to gemmini calls, inline to v2, and hoist all the configurations
    gemmini = set_memory(gemmini, res_alloc, GEMM_ACCUM)
    gemmini = set_memory(gemmini, a_alloc, GEMM_SCRATCH)
    gemmini = set_memory(gemmini, b_alloc, GEMM_SCRATCH)
    # print("10: After setting memories")
    # print(gemmini)

    tuples = [
        # (zero_acc_i32, zero_acc_i32_v2),
        (ld_acc_i32_vector, ld_acc_i32_vector_v2),
        (ld_i8_block_id1, ld_i8_block_id1_v2),
        (ld_i8_block_id2, ld_i8_block_id2_v2),
        (matmul_acc_i8, matmul_acc_i8_v2),
        (st_acc_i8, st_acc_i8_v2),
    ]
    for t in tuples:
        gemmini = simplify(replace_and_inline(gemmini, t))
    # print("11: After replacing and inlining")
    # print(gemmini)
    # # Add a guard to redundant loads
    gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id1(_)"))
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
    return cpu, gemmini


def test_matmul_all_woFusion(NN, MM, KK):
    # KK = 512
    assert NN % 64 == 0, "N must be a multiple of 64"
    assert MM % 64 == 0, "M must be a multiple of 64"
    assert KK % 64 == 0, "K must be a multiple of 64"
    cpu = rename(
        matmul_algorithm(), "matmul_on_cpu"
    )  # Rename "matmul" to "matmul_on_cpu"
    cpu = cpu.partial_eval(K=KK, N=NN, M=MM)

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
    gemmini, _ = tile_loops(gemmini, [(i_loop, 16), (j_loop, 16)], perfect=True)
    # print(i_loop)
    if NN % 512 == 0 and MM % 512 == 0:
        gemmini, [_, j_outer] = tile_loops(
            gemmini, [(i_loop, 16), (j_loop, 16)], perfect=True
        )

    gemmini, _ = tile_loops(gemmini, [(k_loop, 16)])
    # print(gemmini.forward(res_alloc))
    # # Bind and lift scratchpad & accumulator memories
    print("gemmini before autolift")
    print(gemmini)

    gemmini = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
    gemmini, a_load, a_alloc = bind_and_lift(
        gemmini, gemmini.forward(a_assign).rhs(), max_size=sc_size / 2
    )
    gemmini, b_load, b_alloc = bind_and_lift(
        gemmini, gemmini.forward(b_assign).rhs(), max_size=sc_size / 2
    )
    print("after autolift")
    print(gemmini)
    # # Divide by 4 to use load_blocks
    gemmini, _ = tile_loops(gemmini, [(k_loop, 4)])
    if NN % 512 == 0 and MM % 512 == 0:
        gemmini, [j_imost] = tile_loops(gemmini, [(j_outer, 4)], perfect=True)
    else:
        gemmini, [j_imost] = tile_loops(gemmini, [(j_loop, 4)], perfect=True)

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
    # print("8: After fissioning")
    # print(gemmini)

    # # Fix indexing
    gemmini = rearrange_dim(gemmini, a_alloc, [0, 2, 1, 3])
    gemmini = rearrange_dim(gemmini, b_alloc, [2, 0, 3, 1])
    gemmini = reorder_loops_from_idx(gemmini, a_load)
    gemmini = reorder_loops_from_idx(gemmini, b_load)
    gemmini = reorder_loops_from_idx(
        gemmini, gemmini.forward(a_assign).rhs()
    )  # you don't need to do this, it seems to be done automatically
    # print("gemmini after rearranging dimensions")
    # print(gemmini)
    gemmini = remove_redundant_loops(gemmini, a_load, num=2)
    gemmini = remove_redundant_loops(gemmini, b_load, num=1)
    # print("9: After rearranging dimensions")
    # print(gemmini)

    # # Replace to gemmini calls, inline to v2, and hoist all the configurations
    gemmini = set_memory(gemmini, res_alloc, GEMM_ACCUM)
    gemmini = set_memory(gemmini, a_alloc, GEMM_SCRATCH)
    gemmini = set_memory(gemmini, b_alloc, GEMM_SCRATCH)
    print("10: After setting memories")
    print(gemmini)

    tuples = [
        # (zero_acc_i32, zero_acc_i32_v2),
        (ld_acc_i32_vector, ld_acc_i32_vector_v2),
        (ld_i8_block_id1, ld_i8_block_id1_v2),
        (ld_i8_block_id2, ld_i8_block_id2_v2),
        (matmul_acc_i8, matmul_acc_i8_v2),
        (st_acc_i8, st_acc_i8_v2),
    ]
    for t in tuples:
        gemmini = simplify(replace_and_inline(gemmini, t))
    # print("11: After replacing and inlining")
    # print(gemmini)
    # # Add a guard to redundant loads
    gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id1(_)"))
    gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id2(_)"))
    print("12: After adding guards")
    print(gemmini)
    # print(gemmini.find_loop("io"))
    # # fuse loops, and unroll
    # gemmini = fuse_all_loops(gemmini, gemmini.body()[0])
    gemmini, _ = fuse_two_loops(gemmini, gemmini.find_loop("io"))
    gemmini, _ = fuse_two_loops(gemmini, gemmini.find_loop("io"))
    gemmini, _ = fuse_two_loops(gemmini, gemmini.find_loop("io"))
    gemmini, _ = fuse_two_loops(gemmini, gemmini.find_loop("io"))
    gemmini, _ = fuse_two_loops(gemmini, gemmini.find_loop("joo"))
    gemmini, _ = fuse_two_loops(gemmini, gemmini.find_loop("joo"))
    gemmini, _ = fuse_two_loops(gemmini, gemmini.find_loop("joo"))
    gemmini, _ = fuse_two_loops(gemmini, gemmini.find_loop("joo"))

    # print("13: After fusing all loops")
    # print(gemmini)
    # gemmini = unroll_all(gemmini, gemmini.find_loop(j_imost.name(), many=True))
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
    # print(gemmini.c_code_str())
    return cpu, gemmini


def test_matmul_explore_1(NN, MM, KK):
    # KK = 512
    assert NN % 64 == 0, "N must be a multiple of 64"
    assert MM % 64 == 0, "M must be a multiple of 64"
    assert KK % 64 == 0, "K must be a multiple of 64"
    cpu = rename(
        matmul_algorithm(), "matmul_on_cpu"
    )  # Rename "matmul" to "matmul_on_cpu"
    cpu = cpu.partial_eval(K=KK, N=NN, M=MM)

    # cpu = cpu.add_assertion("N % 256 == 0")
    # cpu = cpu.add_assertion("M % 256 == 0")

    # Rename the procedure to "matmul_on_gemmini"
    gemmini = rename(cpu, "matmul_on_gemmini")

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

    # Schedule starts here!!

    # Tile loops for a scratchpad and an accumulator
    gemmini, _ = tile_loops(gemmini, [(i_loop, 16), (j_loop, 16)], perfect=True)
    # print(i_loop)
    if NN % 512 == 0 and MM % 512 == 0:
        gemmini, [_, j_outer] = tile_loops(
            gemmini, [(i_loop, 16), (j_loop, 16)], perfect=True
        )

    gemmini, _ = tile_loops(gemmini, [(k_loop, 16)])
    # print(gemmini.forward(res_alloc))
    # # Bind and lift scratchpad & accumulator memories
    gemmini = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
    gemmini, a_load, a_alloc = bind_and_lift(
        gemmini, gemmini.forward(a_assign).rhs(), max_size=sc_size / 2
    )
    gemmini, b_load, b_alloc = bind_and_lift(
        gemmini, gemmini.forward(b_assign).rhs(), max_size=sc_size / 2
    )
    # print("gemmini before third tiling")
    # print(gemmini)
    # # Divide by 4 to use load_blocks
    gemmini, _ = tile_loops(gemmini, [(k_loop, 4)])
    if NN % 512 == 0 and MM % 512 == 0:
        gemmini, [j_imost] = tile_loops(gemmini, [(j_outer, 4)], perfect=True)
    else:
        gemmini, [j_imost] = tile_loops(gemmini, [(j_loop, 4)], perfect=True)

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
    gemmini = rearrange_dim(gemmini, a_alloc, [0, 2, 1, 3])
    gemmini = rearrange_dim(gemmini, b_alloc, [2, 0, 3, 1])
    gemmini = reorder_loops_from_idx(gemmini, a_load)
    gemmini = reorder_loops_from_idx(gemmini, b_load)
    gemmini = reorder_loops_from_idx(
        gemmini, gemmini.forward(a_assign).rhs()
    )  # you don't need to do this, it seems to be done automatically
    print("gemmini after rearranging dimensions")
    print(gemmini)
    gemmini = remove_redundant_loops(gemmini, a_load, num=3)
    gemmini = remove_redundant_loops(gemmini, b_load, num=2)
    # print("9: After rearranging dimensions")
    # print(gemmini)

    # # Replace to gemmini calls, inline to v2, and hoist all the configurations
    gemmini = set_memory(gemmini, res_alloc, GEMM_ACCUM)
    gemmini = set_memory(gemmini, a_alloc, GEMM_SCRATCH)
    gemmini = set_memory(gemmini, b_alloc, GEMM_SCRATCH)
    # print("10: After setting memories")
    # print(gemmini)

    tuples = [
        # (zero_acc_i32, zero_acc_i32_v2),
        (ld_acc_i32_vector, ld_acc_i32_vector_v2),
        (ld_i8_block_id1, ld_i8_block_id1_v2),
        (ld_i8_block_id2, ld_i8_block_id2_v2),
        (matmul_acc_i8, matmul_acc_i8_v2),
        (st_acc_i8, st_acc_i8_v2),
    ]
    for t in tuples:
        gemmini = simplify(replace_and_inline(gemmini, t))
    # print("11: After replacing and inlining")
    # print(gemmini)
    # # Add a guard to redundant loads
    gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id1(_)"))
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
    # print(gemmini.c_code_str())
    return cpu, gemmini


def test_matmul_autoTune(NN, MM, KK):
    # KK = 512
    # assert NN % 64 == 0, "N must be a multiple of 64"
    # assert MM % 64 == 0, "M must be a multiple of 64"
    # assert KK % 64 == 0, "K must be a multiple of 64"
    cpu = rename(
        matmul_algorithm(), "matmul_on_cpu"
    )  # Rename "matmul" to "matmul_on_cpu"
    cpu = cpu.partial_eval(K=KK, N=NN, M=MM)

    # cpu = cpu.add_assertion("N % 256 == 0")
    # cpu = cpu.add_assertion("M % 256 == 0")

    # Rename the procedure to "matmul_on_gemmini"
    gemmini = rename(cpu, "matmul_on_gemmini")

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

    # Schedule starts here!!

    # Tile loops for a scratchpad and an accumulator
    gemmini, _ = tile_loops(gemmini, [(i_loop, 16), (j_loop, 16)], perfect=True)
    # print(i_loop)
    if NN % 512 == 0 and MM % 512 == 0:
        gemmini, [_, j_outer] = tile_loops(
            gemmini, [(i_loop, 16), (j_loop, 16)], perfect=True
        )

    gemmini, _ = tile_loops(gemmini, [(k_loop, 16)])
    # print(gemmini.forward(res_alloc))
    # # Bind and lift scratchpad & accumulator memories
    gemmini = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
    gemmini, a_load, a_alloc = bind_and_lift(
        gemmini, gemmini.forward(a_assign).rhs(), max_size=sc_size / 2
    )
    gemmini, b_load, b_alloc = bind_and_lift(
        gemmini, gemmini.forward(b_assign).rhs(), max_size=sc_size / 2
    )
    # print("gemmini before third tiling")
    # print(gemmini)
    # # Divide by 4 to use load_blocks
    gemmini, _ = tile_loops(gemmini, [(k_loop, 4)])
    if NN % 512 == 0 and MM % 512 == 0:
        gemmini, [j_imost] = tile_loops(gemmini, [(j_outer, 4)], perfect=True)
    else:
        gemmini, [j_imost] = tile_loops(gemmini, [(j_loop, 4)], perfect=True)

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
    gemmini = rearrange_dim(gemmini, a_alloc, [0, 2, 1, 3])
    gemmini = rearrange_dim(gemmini, b_alloc, [2, 0, 3, 1])
    gemmini = reorder_loops_from_idx(gemmini, a_load)
    gemmini = reorder_loops_from_idx(gemmini, b_load)
    gemmini = reorder_loops_from_idx(
        gemmini, gemmini.forward(a_assign).rhs()
    )  # you don't need to do this, it seems to be done automatically
    print("gemmini after rearranging dimensions")
    print(gemmini)
    gemmini = remove_redundant_loops(gemmini, a_load, num=3)
    gemmini = remove_redundant_loops(gemmini, b_load, num=2)
    # print("9: After rearranging dimensions")
    # print(gemmini)

    # # Replace to gemmini calls, inline to v2, and hoist all the configurations
    gemmini = set_memory(gemmini, res_alloc, GEMM_ACCUM)
    gemmini = set_memory(gemmini, a_alloc, GEMM_SCRATCH)
    gemmini = set_memory(gemmini, b_alloc, GEMM_SCRATCH)
    # print("10: After setting memories")
    # print(gemmini)

    tuples = [
        # (zero_acc_i32, zero_acc_i32_v2),
        (ld_acc_i32_vector, ld_acc_i32_vector_v2),
        (ld_i8_block_id1, ld_i8_block_id1_v2),
        (ld_i8_block_id2, ld_i8_block_id2_v2),
        (matmul_acc_i8, matmul_acc_i8_v2),
        (st_acc_i8, st_acc_i8_v2),
    ]
    for t in tuples:
        gemmini = simplify(replace_and_inline(gemmini, t))
    # print("11: After replacing and inlining")
    # print(gemmini)
    # # Add a guard to redundant loads
    gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id1(_)"))
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
    # print(gemmini.c_code_str())
    return cpu, gemmini


def test_matmul_cosa():
    KK = 512
    NN = 512
    MM = 512
    # assert NN % 64 == 0, "N must be a multiple of 64"
    # assert MM % 64 == 0, "M must be a multiple of 64"
    # assert KK % 64 == 0, "K must be a multiple of 64"
    cpu = rename(
        matmul_algorithm(), "matmul_on_cpu"
    )  # Rename "matmul" to "matmul_on_cpu"
    cpu = cpu.partial_eval(K=KK, N=NN, M=MM)

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
    gemmini, [ii, ji] = tile_loops(gemmini, [(i_loop, 16), (j_loop, 16)], perfect=True)
    # gemmini, _ = tile_loops(gemmini, [(k_loop, 16)])
    # gemmini = lift_scope_n(gemmini, k_loop, n_lifts=2)

    gemmini, _ = tile_loops(gemmini, [(i_loop, 8), (j_loop, 8)], perfect=True)

    gemmini = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
    gemmini = fission_as_much_as_possible(gemmini, res_load)
    print(gemmini)

    gemmini, _ = tile_loops(gemmini, [(ii, 1), (ji, 1), (k_loop, 16)])
    gemmini, _ = tile_loops(gemmini, [(ii, 1), (ji, 1), (k_loop, 4)])
    # print(gemmini)
    # # gemmini, _ = tile_loops(gemmini, [(i_loop, 4), (j_loop, 4)])
    gemmini, _ = tile_loops(gemmini, [(k_loop, 4)])
    # # print(gemmini.forward(i_loop), gemmini.forward(j_loop))
    print(gemmini)

    # gemmini = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
    # gemmini, a_load, a_alloc = bind_and_lift(
    #     gemmini, gemmini.forward(a_assign).rhs(), max_size=sc_size/2
    # )
    # gemmini, b_load, b_alloc = bind_and_lift(
    #     gemmini, gemmini.forward(b_assign).rhs(), max_size=sc_size/2
    # )
    # print("after autolift")
    # print(gemmini)
    # # Divide by 4 to use load_blocks
    # gemmini, _ = tile_loops(gemmini, [(k_loop, 4)])
    # if NN % 512 == 0 and MM % 512 == 0:
    #     gemmini, [j_imost] = tile_loops(gemmini, [(j_outer, 4)], perfect=True)
    # else:
    #     gemmini, [j_imost] = tile_loops(gemmini, [(j_loop, 4)], perfect=True)

    # # # Fission all the loops
    # gemmini = fission_as_much_as_possible(gemmini, res_load)
    # # print("5: After fissioning res_load")
    # # print(gemmini)
    # gemmini = fission_as_much_as_possible(gemmini, k_loop)
    # # print("6: After fissioning k_loop")
    # # print(gemmini)
    # gemmini = fission_as_much_as_possible(gemmini, a_load)
    # # print("7: After fissioning a_load")
    # # print(gemmini)
    # gemmini = fission_as_much_as_possible(gemmini, b_load)
    # # print("8: After fissioning")
    # # print(gemmini)

    # # # Fix indexing
    # gemmini = rearrange_dim(gemmini, a_alloc, [0, 2, 1, 3])
    # gemmini = rearrange_dim(gemmini, b_alloc, [2, 0, 3, 1])
    # gemmini = reorder_loops_from_idx(gemmini, a_load)
    # gemmini = reorder_loops_from_idx(gemmini, b_load)
    # gemmini = reorder_loops_from_idx(gemmini, gemmini.forward(a_assign).rhs()) #you don't need to do this, it seems to be done automatically
    # # print("gemmini after rearranging dimensions")
    # # print(gemmini)
    # gemmini = remove_redundant_loops(gemmini, a_load, num=2)
    # gemmini = remove_redundant_loops(gemmini, b_load, num=1)
    # # print("9: After rearranging dimensions")
    # # print(gemmini)

    # # # Replace to gemmini calls, inline to v2, and hoist all the configurations
    # gemmini = set_memory(gemmini, res_alloc, GEMM_ACCUM)
    # gemmini = set_memory(gemmini, a_alloc, GEMM_SCRATCH)
    # gemmini = set_memory(gemmini, b_alloc, GEMM_SCRATCH)
    # # print("10: After setting memories")
    # # print(gemmini)

    # tuples = [
    #     # (zero_acc_i32, zero_acc_i32_v2),
    #     (ld_acc_i32_vector, ld_acc_i32_vector_v2),
    #     (ld_i8_block_id1, ld_i8_block_id1_v2),
    #     (ld_i8_block_id2, ld_i8_block_id2_v2),
    #     (matmul_acc_i8, matmul_acc_i8_v2),
    #     (st_acc_i8, st_acc_i8_v2),
    # ]
    # for t in tuples:
    #     gemmini = simplify(replace_and_inline(gemmini, t))
    # # print("11: After replacing and inlining")
    # # print(gemmini)
    # # # Add a guard to redundant loads
    # gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id1(_)"))
    # gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id2(_)"))
    # # print("12: After adding guards")
    # # print(gemmini)

    # # # fuse loops, and unroll
    # gemmini = fuse_all_loops(gemmini, gemmini.body()[0])
    # # print("13: After fusing all loops")
    # # print(gemmini)
    # gemmini = unroll_all(gemmini, gemmini.find_loop(j_imost.name(), many=True))
    # # print("14: After unrolling j_imost")
    # # print(gemmini)
    # # # Schedule ends here!!
    # # # 29 lines excluding comments and newlines

    # gemmini = simplify(gemmini) #mathematical simplifications 2*16 -> 32, etc.

    # print("")
    # print("============= THIS IS THE SCHEDULED MATMUL ===============")
    # print(gemmini)
    # print("============= THIS IS THE SCHEDULED MATMUL ===============")
    # print("")
    return cpu, gemmini


# test_matmul()
# test_matmul_128()
# test_matmul_256()
# test_matmul_64()
test_matmul_all(64, 64, 64)
# test_matmul_all(128, 128, 128)
# test_matmul_all(256, 256, 256)
# test_matmul_all(512, 512, 512)
# test_matmul_explore_1(128, 128, 128)


### test without fusion
# test_matmul_all_woFusion(64, 64, 64)
# test_matmul_all_woFusion(128, 128, 128)
# test_matmul_all_woFusion(256, 256, 256)
# test_matmul_all_woFusion(512, 512, 512)
