from __future__ import annotations

from exo.platforms.gemmini import (
    # ld_i8_block_id2_new,
    # ld_i8_block_id2_v2_new,
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
    gemmini, _ = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
    (gemmini, _), a_load, a_alloc = bind_and_lift(
        gemmini, gemmini.forward(a_assign).rhs(), max_size=sc_size
    )
    (gemmini, _), b_load, b_alloc = bind_and_lift(
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
    gemmini, _ = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
    (gemmini, _), a_load, a_alloc = bind_and_lift(
        gemmini, gemmini.forward(a_assign).rhs(), max_size=sc_size
    )
    (gemmini, _), b_load, b_alloc = bind_and_lift(
        gemmini, gemmini.forward(b_assign).rhs(), max_size=sc_size
    )
    print(f"a_load: {a_load}")
    # print("gemmini before third tiling")
    # print(gemmini)
    # # Divide by 4 to use load_blocks
    gemmini, _ = tile_loops(gemmini, [(k_loop, 4)])
    gemmini, [j_imost] = tile_loops(gemmini, [(j_outer, 4)], perfect=True)
    # print("before fissioning")
    # print(gemmini)
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
    # assert gemmini.c_code_str() == golden


def test_matmul_exo_sample2():
    KK = 512
    M = 512
    N = 512
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
    gemmini, _ = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
    (gemmini, _), a_load, a_alloc = bind_and_lift(
        gemmini, gemmini.forward(a_assign).rhs(), max_size=sc_size / 2
    )
    (gemmini, _), b_load, b_alloc = bind_and_lift(
        gemmini, gemmini.forward(b_assign).rhs(), max_size=sc_size / 2
    )
    # print(f"a_load: {a_load}")
    # print("gemmini before third tiling")
    # print(gemmini)
    # # Divide by 4 to use load_blocks
    gemmini, _ = tile_loops(gemmini, [(k_loop, 4)])
    gemmini, [j_imost] = tile_loops(gemmini, [(j_outer, 4)], perfect=True)
    # print("before fissioning")
    # print(gemmini)
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
    # gemmini = rearrange_dim(gemmini, a_alloc, [0, 1, 3, 2, 4])
    gemmini = rearrange_dim(gemmini, b_alloc, [2, 0, 3, 1])
    gemmini = reorder_loops_from_idx(gemmini, a_load)
    gemmini = reorder_loops_from_idx(gemmini, b_load)
    gemmini = reorder_loops_from_idx(
        gemmini, gemmini.forward(a_assign).rhs()
    )  # you don't need to do this, it seems to be done automatically
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
    print("12: After adding guards")
    print(gemmini)

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


def test_matmul_exo_sample3():
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
    gemmini, _ = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
    (gemmini, _), a_load, a_alloc = bind_and_lift(
        gemmini, gemmini.forward(a_assign).rhs(), max_size=sc_size
    )
    (gemmini, _), b_load, b_alloc = bind_and_lift(
        gemmini, gemmini.forward(b_assign).rhs(), max_size=sc_size
    )
    # print(f"a_load: {a_load}")
    # print("gemmini before third tiling")
    # print(gemmini)
    # # Divide by 4 to use load_blocks
    gemmini, _ = tile_loops(gemmini, [(k_loop, 4)])
    gemmini, [j_imost] = tile_loops(gemmini, [(j_outer, 4)], perfect=True)
    # print("before fissioning")
    # print(gemmini)
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
    # gemmini = rearrange_dim(gemmini, a_alloc, [0, 2, 1, 3])
    gemmini = rearrange_dim(gemmini, a_alloc, [0, 1, 3, 2, 4])
    gemmini = rearrange_dim(gemmini, b_alloc, [2, 0, 3, 1])
    gemmini = reorder_loops_from_idx(gemmini, a_load)
    gemmini = reorder_loops_from_idx(gemmini, b_load)
    gemmini = reorder_loops_from_idx(
        gemmini, gemmini.forward(a_assign).rhs()
    )  # you don't need to do this, it seems to be done automatically
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
    print("12: After adding guards")
    print(gemmini)

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


def test_matmul_exo_sample1():
    KK = 512
    M = 256
    N = 1024
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
    gemmini, _ = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
    (gemmini, _), a_load, a_alloc = bind_and_lift(
        gemmini, gemmini.forward(a_assign).rhs(), max_size=sc_size
    )
    (gemmini, _), b_load, b_alloc = bind_and_lift(
        gemmini, gemmini.forward(b_assign).rhs(), max_size=sc_size
    )
    # print(f"a_load: {a_load}")
    # print("gemmini before third tiling")
    # print(gemmini)
    # # Divide by 4 to use load_blocks
    gemmini, _ = tile_loops(gemmini, [(k_loop, 4)])
    gemmini, [j_imost] = tile_loops(gemmini, [(j_outer, 4)], perfect=True)
    # print("before fissioning")
    # print(gemmini)
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
    gemmini = rearrange_dim(gemmini, b_alloc, [0, 3, 1, 4, 2])
    # gemmini = rearrange_dim(gemmini, b_alloc, [2, 0, 3, 1])
    gemmini = reorder_loops_from_idx(gemmini, a_load)
    gemmini = reorder_loops_from_idx(gemmini, b_load)
    gemmini = reorder_loops_from_idx(
        gemmini, gemmini.forward(a_assign).rhs()
    )  # you don't need to do this, it seems to be done automatically
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
    print("12: After adding guards")
    print(gemmini)

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


def test_matmul_zz(N, M):
    KK = 512

    cpu = rename(
        matmul_algorithm(), "matmul_on_cpu"
    )  # Rename "matmul" to "matmul_on_cpu"
    cpu = cpu.partial_eval(N=N, M=M, K=KK)
    # cpu = cpu.add_assertion("N % 256 == 0")
    # cpu = cpu.add_assertion("M % 256 == 0")
    assert N % 256 == 0, "N must be a multiple of 256"
    assert M % 256 == 0, "M must be a multiple of 256"

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
    # Tile loops based on a mapping loop vals: (i, iv0, iv1, iv2), (j, jv0, jv1, jv2), (k, kv0, kv1, kv2) and loop orders (i,j,k) -> (iv0, jv0, kv0, iv1, jv1, kv1, iv2, jv2, kv2)
    gemmini = divide_loop(gemmini, i_loop, 64, ["io", "ii"], perfect=True)
    gemmini = divide_loop(gemmini, j_loop, 16, ["jo", "ji"], perfect=True)
    gemmini = divide_loop(gemmini, "for jo in _:_", 4, ["joo", "joi"], perfect=True)
    gemmini = divide_loop(gemmini, k_loop, 16, ["ko", "ki"], perfect=True)
    gemmini = divide_loop(gemmini, "for ko in _:_", 8, ["koo", "koi"], perfect=True)

    gemmini = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
    gemmini, a_load, a_alloc = bind_and_lift(
        gemmini, gemmini.forward(a_assign).rhs(), max_size=sc_size
    )
    gemmini, b_load, b_alloc = bind_and_lift(
        gemmini, gemmini.forward(b_assign).rhs(), max_size=sc_size
    )

    gemmini = fission_as_much_as_possible(gemmini, res_load)
    gemmini = fission_as_much_as_possible(gemmini, k_loop)
    gemmini = fission_as_much_as_possible(gemmini, a_load)
    gemmini = fission_as_much_as_possible(gemmini, b_load)

    gemmini = rearrange_dim(gemmini, a_alloc, [0, 2, 1, 3])
    gemmini = rearrange_dim(gemmini, b_alloc, [2, 0, 3, 1])
    gemmini = reorder_loops_from_idx(gemmini, a_load)
    gemmini = reorder_loops_from_idx(gemmini, b_load)
    # gemmini = reorder_loops_from_idx_zz(gemmini, a_load, ["io", "koo", "koi", "joo", "joi", "ii", "ji", "ki"])
    print(gemmini)
    # gemmini = reorder_loops(gemmini, 'ji koo')
    # gemmini = reorder_mapping(gemmini, ["io", "ii", "joo", "joi", "ji", "koo", "koi", "ki"], ["io", "koo", "koi", "joo", "joi", "ii", "ji", "ki"])


def reorder_mapping(proc, original, target):
    pos_in_target = {loop: i for i, loop in enumerate(target)}

    # Build the permutation: perm[i] = where original[i] must go
    perm = [pos_in_target[x] for x in original]

    n = len(original)
    visited = [False] * n
    cycles = []

    for i in range(n):
        if visited[i]:
            continue

        cycle = []
        j = i
        while not visited[j]:
            visited[j] = True
            cycle.append(original[j])
            j = perm[j]

        # Only keep cycles longer than 1 (non-trivial)
        if len(cycle) > 1:
            cycles.append(cycle)
    print("Reordering cycles:")
    print(cycles)
    for cycle in cycles:
        for i in range(len(cycle) - 1):
            print(f"{cycle[i]} {cycle[i+1]}")
            proc = reorder_loops(proc, f"{cycle[i]} {cycle[i+1]}")
            print(proc)


def test_matmul_zz_sample1_optimized():
    KK = 512
    N = 1024
    M = 256
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
    gemmini, _ = tile_loops(gemmini, [(i_loop, 16), (j_loop, 16)], perfect=True)
    gemmini = divide_loop(gemmini, k_loop, 16, ["ko", "ki"], perfect=True)
    # print(gemmini)
    # Bind and lift scratchpad & accumulator memories
    gemmini, _ = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
    (gemmini, acc_size), a_load, a_alloc = bind_and_lift(
        gemmini, gemmini.forward(a_assign).rhs(), max_size=sc_size
    )
    (gemmini, _), b_load, b_alloc = bind_and_lift(
        gemmini, gemmini.forward(b_assign).rhs(), max_size=sc_size - acc_size
    )

    # Divide by 4 to use load_blocks
    # gemmini, _ = tile_loops(gemmini, [(k_loop, 4)])
    # gemmini, [j_imost] = tile_loops(gemmini, [(j_loop, 4)], perfect=True)

    # print("before fissioning")
    # print(gemmini)

    # Fission all the loops
    gemmini = fission_as_much_as_possible(gemmini, res_load)
    # print("5: After fissioning res_load")
    # print(gemmini)
    gemmini = fission_as_much_as_possible(gemmini, b_load)
    # print("7: After fissioning b_load")
    # print(gemmini)
    src_tmp = gemmini.find("src_tmp = res[_]")
    cur = src_tmp.prev().prev()
    gemmini = fission_as_much_as_possible(gemmini, cur)
    # print("6: After fissioning k_loop")
    # print(gemmini)
    gemmini = fission_as_much_as_possible(gemmini, a_load)
    # print("8: After fissioning")
    # print(gemmini)

    # # # Fix indexing
    # gemmini = rearrange_dim(gemmini, a_alloc, [0, 2, 1, 3])
    gemmini = rearrange_dim(gemmini, a_alloc, [1, 0, 2])
    gemmini = rearrange_dim(gemmini, b_alloc, [2, 0, 3, 1])
    gemmini = reorder_loops_from_idx(gemmini, a_load)
    gemmini = reorder_loops_from_idx(gemmini, b_load)
    # print(gemmini.forward(a_assign))
    gemmini = reorder_loops_from_idx(
        gemmini, gemmini.forward(a_assign).rhs()
    )  # you don't need to do this, it seems to be done automatically
    gemmini = remove_redundant_loops(
        gemmini, a_load, num=2
    )  # original 2: check for final zizag mapping (fusion)
    gemmini = remove_redundant_loops(gemmini, b_load, num=2)

    gemmini = divide_loop(
        gemmini,
        gemmini.forward(a_load).parent().parent().parent(),
        4,
        ["koo", "koi"],
        perfect=True,
    )
    gemmini = divide_loop(
        gemmini,
        gemmini.forward(b_load).parent().parent().parent(),
        4,
        ["joo", "joi"],
        perfect=True,
    )
    # print("9: After rearranging dimensions")
    # print(gemmini)

    # Replace to gemmini calls, inline to v2, and hoist all the configurations
    gemmini = set_memory(gemmini, res_alloc, GEMM_ACCUM)
    gemmini = set_memory(gemmini, a_alloc, GEMM_SCRATCH)
    gemmini = set_memory(gemmini, b_alloc, GEMM_SCRATCH)
    print("10: After setting memories")
    print(gemmini)

    tuples = [
        # (zero_acc_i32, zero_acc_i32_v2),
        (ld_acc_i32_vector, ld_acc_i32_vector_v2),
        (ld_i8_block_id2, ld_i8_block_id2_v2),
        (ld_i8_block_id1, ld_i8_block_id1_v2),
        (matmul_acc_i8, matmul_acc_i8_v2),
        (st_acc_i8, st_acc_i8_v2),
    ]
    for t in tuples:
        gemmini = simplify(replace_and_inline(gemmini, t))
    # print("11: After replacing and inlining")
    # print(gemmini)
    # Add a guard to redundant loads
    gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id1(_)"))
    gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id2(_)"))
    print("12: After adding guards")
    print(gemmini)

    # fuse loops, and unroll
    gemmini = reorder_top(gemmini, gemmini.find_loop("ko"))
    # gemmini, _ = fuse_two_loops(gemmini, gemmini.find_loop("io"))
    # gemm_k_loop = gemmini.find_loop("do_matmul_acc_i8(_)").parent().parent()
    # gemmini = mult_loops(gemmini, gemm_k_loop, "ko")
    # gemm_j_loop = gemm_k_loop.parent().parent()
    # gemmini = mult_loops(gemmini, gemm_j_loop, "jo")
    gemmini = reorder_loops(gemmini, "jo ko")
    gemmini = fuse_all_loops(gemmini, gemmini.body()[0])

    # gemmini = mult_loops(gemmini, gemmini.find_loop("koo"), 'ko')

    # gemmini = unroll_all(gemmini, gemmini.find_loop(j_imost.name(), many=True))
    # # print("14: After unrolling j_imost")
    # # print(gemmini)
    # Schedule ends here!!
    # # # 29 lines excluding comments and newlines

    gemmini = simplify(gemmini)  # mathematical simplifications 2*16 -> 32, etc.
    print(gemmini)
    print(gemmini.c_code_str())


def test_matmul_zz_sample2():
    KK = 512
    N = 512
    M = 512
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
    gemmini = stage_mem(gemmini, "for i in _:_", "A[0:256, 0:512]", "A_temp")
    # gemmini = divide_dim(gemmini, gemmini.find("A_temp: _"), 1, 16)
    # gemmini = rearrange_dim(gemmini, gemmini.find("A_temp: _"), [1, 0, 2])
    # print(gemmini)
    # i0_loop = gemmini.find_loop("i0")
    # i1_loop = gemmini.find_loop("i1")
    # gemmini, _ = tile_loops(gemmini, [(i0_loop, 16), (i1_loop, 16)], perfect=True)
    # gemmini = divide_loop(gemmini, i1_loop, 4, ["i1oo", "i1oi"], perfect=True)
    # print(gemmini)

    # Tile loops for a scratchpad and an accumulator
    # gemmini = divide_loop(gemmini, i_loop, 64, ["io", "ii"], perfect=True)
    # gemmini = divide_loop(gemmini, j_loop, 16, ["jo", "ji"], perfect=True)
    gemmini = reorder_loops(gemmini, "i j")
    gemmini, _ = tile_loops(gemmini, [(j_loop, 16), (i_loop, 16)], perfect=True)
    # gemmini, _ = tile_loops(gemmini, [(i_loop, 16), (j_loop, 16)], perfect=True)
    gemmini = divide_loop(gemmini, k_loop, 16, ["ko", "ki"], perfect=True)
    # Bind and lift scratchpad & accumulator memories
    gemmini = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
    # gemmini = lift_scope(gemmini, gemmini.forward(k_loop))
    # gemmini = reorder_loops(gemmini, "ii ko")
    # gemmini, a_load, a_alloc = bind_and_lift(
    #     gemmini, gemmini.forward(a_assign).rhs(), max_size=sc_size
    # )
    # gemmini, b_load, b_alloc = bind_and_lift(
    #     gemmini, gemmini.forward(b_assign).rhs(), max_size=sc_size
    # )
    # print(f"a_load: {a_load}")
    # print(f"a_alloc: {a_alloc}")
    # print("gemmini before third tiling")
    # print(gemmini)
    # Divide by 4 to use load_blocks
    gemmini, _ = tile_loops(gemmini, [(k_loop, 4)])
    gemmini, [j_imost] = tile_loops(gemmini, [(j_loop, 4)], perfect=True)
    # # print("before fissioning")
    # # print(gemmini)
    # # # Fission all the loops
    gemmini = fission_as_much_as_possible(gemmini, res_load)
    print("5: After fissioning res_load")
    print(gemmini)
    gemmini = fission_as_much_as_possible(gemmini, k_loop)
    gemmini = reorder_loops(gemmini, "ii koo")
    gemmini = reorder_loops(gemmini, "ji koo")
    gemmini = reorder_loops(gemmini, "io koo")
    print(gemmini)
    gemmini = fission_as_much_as_possible(gemmini, k_loop)
    # gemmini = reorder_loops(gemmini, "joi koo")
    # gemmini, a_load, a_alloc = bind_and_lift(
    #     gemmini, gemmini.forward(a_assign).rhs(), max_size=sc_size
    # )
    # gemmini, b_load, b_alloc = bind_and_lift(
    #     gemmini, gemmini.forward(b_assign).rhs(), max_size=sc_size
    # )
    # # print("7: After fissioning b_load")
    # # print(gemmini)
    # src_tmp = gemmini.find("src_tmp = res[_]")
    # cur = src_tmp.prev().prev()
    # gemmini = fission_as_much_as_possible(gemmini, cur)
    # # print("6: After fissioning k_loop")
    # # print(gemmini)
    # gemmini = fission_as_much_as_possible(gemmini, a_load)

    print("8: After fissioning")
    print(gemmini)

    # # # # Fix indexing
    # # gemmini = rearrange_dim(gemmini, a_alloc, [0, 2, 1, 3])
    # gemmini = rearrange_dim(gemmini, a_alloc, [1, 0, 2])
    # gemmini = rearrange_dim(gemmini, b_alloc, [2, 0, 3, 1])
    # gemmini = reorder_loops_from_idx(gemmini, a_load)
    # gemmini = reorder_loops_from_idx(gemmini, b_load)
    # print(gemmini.forward(a_assign))
    # gemmini = reorder_loops_from_idx(
    #     gemmini, gemmini.forward(a_assign).rhs()
    # )  # you don't need to do this, it seems to be done automatically
    # gemmini = remove_redundant_loops(gemmini, a_load, num=2) # original 2: check for final zizag mapping (fusion)
    # gemmini = remove_redundant_loops(gemmini, b_load, num=2)
    # # print("9: After rearranging dimensions")
    # # print(gemmini)

    # # Replace to gemmini calls, inline to v2, and hoist all the configurations
    # gemmini = set_memory(gemmini, res_alloc, GEMM_ACCUM)
    # gemmini = set_memory(gemmini, a_alloc, GEMM_SCRATCH)
    # gemmini = set_memory(gemmini, b_alloc, GEMM_SCRATCH)
    # # print("10: After setting memories")
    # # print(gemmini)

    # tuples = [
    #     # (zero_acc_i32, zero_acc_i32_v2),
    #     (ld_acc_i32_vector, ld_acc_i32_vector_v2),
    #     (ld_i8_block_id2, ld_i8_block_id2_v2),
    #     (ld_i8_block_id1, ld_i8_block_id1_v2),
    #     (matmul_acc_i8, matmul_acc_i8_v2),
    #     (st_acc_i8, st_acc_i8_v2),
    # ]
    # for t in tuples:
    #     gemmini = simplify(replace_and_inline(gemmini, t))
    # # print("11: After replacing and inlining")
    # # print(gemmini)
    # # Add a guard to redundant loads
    # gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id1(_)"))
    # gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id2(_)"))
    # print("12: After adding guards")
    # print(gemmini)

    # # fuse loops, and unroll
    # gemmini = reorder_top(gemmini, gemmini.find_loop("koo"))
    # # gemmini = fuse_all_loops(gemmini, gemmini.body()[0])
    # gemmini, _ = fuse_two_loops(gemmini, gemmini.find_loop("io"))
    # gemm_k_loop = gemmini.find_loop("do_matmul_acc_i8(_)").parent().parent()
    # gemmini = mult_loops(gemmini, gemm_k_loop, "ko")
    # gemm_j_loop = gemm_k_loop.parent().parent()
    # gemmini = mult_loops(gemmini, gemm_j_loop, "jo")
    # gemmini = reorder_loops(gemmini, "jo ko")
    # # gemmini, _ = fuse_two_loops(gemmini, gemmini.find_loop("koo"))
    # # gemmini, _ = fuse_two_loops(gemmini, gemmini.find_loop("joo"))
    # # gemmini, _ = fuse_two_loops(gemmini, gemmini.find_loop("joo"))
    # # gemmini, _ = fuse_two_loops(gemmini, gemmini.find_loop("joo"))
    # print("13: After fusing loops")
    # print(gemmini)


def test_matmul_zz_sample2_optimized():
    KK = 512
    N = 512
    M = 512
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

    gemmini = reorder_loops(gemmini, "i j")
    gemmini, _ = tile_loops(gemmini, [(j_loop, 16), (i_loop, 16)], perfect=True)
    gemmini = divide_loop(gemmini, j_loop, 8, ["joo", "joi"], perfect=True)
    gemmini = reorder_loops(gemmini, "joi io")
    gemmini = reorder_loops(gemmini, "ji ii")
    gemmini = divide_loop(gemmini, k_loop, 16, ["ko", "ki"], perfect=True)
    print(gemmini)

    gemmini, _ = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
    (gemmini, occ_size), b_load, b_alloc = bind_and_lift(
        gemmini, gemmini.forward(b_assign).rhs(), max_size=512 * 128
    )
    (gemmini, _), a_load, a_alloc = bind_and_lift(
        gemmini, gemmini.forward(a_assign).rhs(), max_size=sc_size - occ_size
    )
    # print("after bind and lift b")
    # print(gemmini)
    # gemmini, _ = tile_loops(gemmini, [(k_loop, 4)])
    # gemmini, _ = tile_loops(gemmini, [(gemmini.find_loop("joi"), 4)])
    print(gemmini)

    # Fission all the loops
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

    # Fix indexing
    gemmini = rearrange_dim(gemmini, a_alloc, [1, 0, 2])
    gemmini = rearrange_dim(gemmini, b_alloc, [2, 0, 3, 1])
    gemmini = reorder_loops_from_idx(gemmini, a_load)
    gemmini = reorder_loops_from_idx(gemmini, b_load)

    gemmini = reorder_loops_from_idx(gemmini, gemmini.forward(b_assign).rhs())
    gemmini = reorder_loops(gemmini, gemmini.forward(b_assign).parent().parent())

    gemmini = remove_redundant_loops(
        gemmini, a_load, num=3
    )  # original 2: check for final zizag mapping (fusion)
    gemmini = remove_redundant_loops(gemmini, b_load, num=2)

    gemmini = divide_loop(
        gemmini,
        gemmini.forward(a_load).parent().parent().parent(),
        4,
        ["koo", "koi"],
        perfect=True,
    )
    gemmini = divide_loop(
        gemmini,
        gemmini.forward(b_load).parent().parent().parent(),
        4,
        ["joio", "joii"],
        perfect=True,
    )

    print(gemmini.forward(a_load))

    # Replace to gemmini calls, inline to v2, and hoist all the configurations
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
    print("11: After replacing and inlining")
    print(gemmini)

    # Add a guard to redundant loads
    gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id1(_)"))
    gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id2(_)"))

    # gemmini = reorder_top()
    # fuse loops, and unroll
    # gemmini, _ = fuse_two_loops(gemmini, gemmini.find_loop("joo"))
    # gemmini = mult_loops(gemmini, gemmini.find_loop('joio'), 'joi')
    gemmini = fuse_all_loops(gemmini, gemmini.body()[0])
    gemmini = simplify(gemmini)
    print(gemmini)
    print(gemmini.c_code_str())


def test_matmul_zz_sample3():
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
    # method 1
    # gemmini = divide_loop(gemmini, k_loop, 16, ["ko", "ki"], perfect=True)
    # (gemmini, _), a_load, a_alloc = bind_and_lift(
    #     gemmini, gemmini.forward(a_assign).rhs(), max_size=sc_size
    # )
    # gemmini = fission_as_much_as_possible(gemmini, a_load)
    # gemmini = reorder_top(gemmini, gemmini.forward(k_loop))
    # method 2
    gemmini = stage_mem(gemmini, "for i in _:_", "A[0:256, 0:512]", "A_temp")
    gemmini = simplify(gemmini)
    print(gemmini)
    gemmini = divide_dim(gemmini, gemmini.find("A_temp: _"), 1, 16)
    gemmini = divide_dim(gemmini, gemmini.find("A_temp: _"), 0, 16)
    gemmini = rearrange_dim(gemmini, gemmini.find("A_temp: _"), [0, 2, 1, 3])
    print(gemmini)
    i0_loop = gemmini.find_loop("i0")
    i1_loop = gemmini.find_loop("i1")
    gemmini, _ = tile_loops(gemmini, [(i0_loop, 16), (i1_loop, 16)], perfect=True)
    gemmini = divide_loop(gemmini, i1_loop, 4, ["i1oo", "i1oi"], perfect=True)
    print(gemmini)

    # Tile loops for a scratchpad and an accumulator
    # print(gemmini)
    gemmini = reorder_loops(gemmini, "i j")
    gemmini, _ = tile_loops(gemmini, [(j_loop, 16), (i_loop, 16)], perfect=True)
    gemmini = divide_loop(gemmini, j_loop, 4, ["joo", "joi"], perfect=True)
    print(gemmini)
    gemmini = divide_loop(gemmini, k_loop, 16, ["ko", "ki"], perfect=True)
    gemmini = reorder_loops(gemmini, "ji ii")
    # # Bind and lift scratchpad & accumulator memories
    gemmini, _ = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
    # (gemmini, occ_size),  a_load, a_alloc = bind_and_lift(
    #     gemmini, gemmini.forward(a_assign).rhs(), max_size=sc_size
    # )
    # print(occ_size)
    occ_size = 256 * 512
    (gemmini, _), b_load, b_alloc = bind_and_lift(
        gemmini, gemmini.forward(b_assign).rhs(), max_size=sc_size - occ_size
    )
    print("after bind and lift b")
    print(gemmini)
    # # Divide by 4 to use load_blocks
    gemmini, _ = tile_loops(gemmini, [(k_loop, 4)])
    # gemmini = divide_loop(gemmini, gemmini.find_loop("i0"), 4, ["joo", "joi"], perfect=True)
    # gemmini, [j_imost] = tile_loops(gemmini, [(j_loop, 4)], perfect=True)
    # # # print("before fissioning")
    # # # print(gemmini)
    # Fission all the loops
    gemmini = fission_as_much_as_possible(gemmini, res_load)
    # print("5: After fissioning res_load")
    # print(gemmini)
    gemmini = fission_as_much_as_possible(gemmini, k_loop)
    # print("6: After fissioning k_loop")
    # print(gemmini)
    # gemmini = fission_as_much_as_possible(gemmini, a_load)
    # print("7: After fissioning a_load")
    # print(gemmini)
    gemmini = fission_as_much_as_possible(gemmini, b_load)
    print("8: After fissioning")
    print(gemmini)

    # Fix indexing
    # # gemmini = rearrange_dim(gemmini, res_alloc, [1, 0, 3, 2])
    # gemmini = rearrange_dim(gemmini, a_alloc, [0, 2, 1, 3])
    gemmini = rearrange_dim(gemmini, b_alloc, [2, 0, 3, 1])
    # # gemmini = reorder_loops_from_idx(gemmini, res_load)
    # gemmini = reorder_loops_from_idx(gemmini, a_load)
    gemmini = reorder_loops_from_idx(gemmini, b_load)

    gemmini = reorder_loops_from_idx(gemmini, gemmini.forward(b_assign).rhs())
    gemmini = reorder_loops(gemmini, gemmini.forward(b_assign).parent().parent())
    # print(gemmini)
    # gemmini = remove_redundant_loops(gemmini, a_load, num=2) # original 2: check for final zizag mapping (fusion)
    gemmini = remove_redundant_loops(gemmini, b_load, num=2)
    # print("9: After rearranging dimensions")
    # print(gemmini)

    # Replace to gemmini calls, inline to v2, and hoist all the configurations
    gemmini = set_memory(gemmini, res_alloc, GEMM_ACCUM)
    gemmini = set_memory(gemmini, gemmini.find("A_temp: _"), GEMM_SCRATCH)
    gemmini = set_memory(gemmini, b_alloc, GEMM_SCRATCH)
    print("10: After setting memories")
    print(gemmini)
    print(ld_i8_block_id1)
    tuples = [
        (ld_acc_i32_vector, ld_acc_i32_vector_v2),
        (ld_i8_block_id1, ld_i8_block_id1_v2),
        (ld_i8_block_id2, ld_i8_block_id2_v2),
        (matmul_acc_i8, matmul_acc_i8_v2),
        (st_acc_i8, st_acc_i8_v2),
    ]
    for t in tuples:
        gemmini = simplify(replace_and_inline(gemmini, t))
    print("11: After replacing and inlining")
    print(gemmini)
    # print(gemmini.c_code_str())


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
    # method 1

    # method 2
    # gemmini = stage_mem(gemmini, "for i in _:_", "A[0:256, 0:512]", "A_temp")
    # gemmini = simplify(gemmini)
    # print(gemmini)
    # gemmini = divide_dim(gemmini, gemmini.find("A_temp: _"), 1, 16)
    # gemmini = divide_dim(gemmini, gemmini.find("A_temp: _"), 0, 16)
    # gemmini = rearrange_dim(gemmini, gemmini.find("A_temp: _"), [0, 2, 1, 3])
    # print(gemmini)
    # i0_loop = gemmini.find_loop("i0")
    # i1_loop = gemmini.find_loop("i1")
    # gemmini, _ = tile_loops(gemmini, [(i0_loop, 16), (i1_loop, 16)], perfect=True)
    # gemmini = divide_loop(gemmini, i1_loop, 4, ["i1oo", "i1oi"], perfect=True)
    # print(gemmini)

    # Tile loops for a scratchpad and an accumulator
    # print(gemmini)
    gemmini = reorder_loops(gemmini, "i j")
    gemmini, _ = tile_loops(gemmini, [(j_loop, 16), (i_loop, 16)], perfect=True)
    gemmini = divide_loop(gemmini, j_loop, 4, ["joo", "joi"], perfect=True)
    print(gemmini)
    gemmini = divide_loop(gemmini, k_loop, 16, ["ko", "ki"], perfect=True)
    gemmini = reorder_loops(gemmini, "ji ii")
    # # Bind and lift scratchpad & accumulator memories
    gemmini, _ = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
    (gemmini, occ_size), a_load, a_alloc = bind_and_lift(
        gemmini, gemmini.forward(a_assign).rhs(), max_size=sc_size
    )
    # print(occ_size)
    occ_size = 256 * 512
    (gemmini, _), b_load, b_alloc = bind_and_lift(
        gemmini, gemmini.forward(b_assign).rhs(), max_size=sc_size - occ_size
    )
    print("after bind and lift b")
    print(gemmini)
    # # Divide by 4 to use load_blocks
    # gemmini, _ = tile_loops(gemmini, [(k_loop, 4)])

    # # # print("before fissioning")
    # # # print(gemmini)
    # Fission all the loops
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

    # Fix indexing
    # # gemmini = rearrange_dim(gemmini, res_alloc, [1, 0, 3, 2])
    gemmini = rearrange_dim(gemmini, a_alloc, [0, 2, 1, 3])
    gemmini = rearrange_dim(gemmini, b_alloc, [2, 0, 3, 1])
    # # gemmini = reorder_loops_from_idx(gemmini, res_load)
    gemmini = reorder_loops_from_idx(gemmini, a_load)
    gemmini = reorder_loops_from_idx(gemmini, b_load)

    gemmini = reorder_loops_from_idx(gemmini, gemmini.forward(b_assign).rhs())
    gemmini = reorder_loops(gemmini, gemmini.forward(b_assign).parent().parent())
    # print(gemmini)
    gemmini = remove_redundant_loops(
        gemmini, a_load, num=2
    )  # original 2: check for final zizag mapping (fusion)
    gemmini = remove_redundant_loops(gemmini, b_load, num=2)

    gemmini = divide_loop(
        gemmini,
        gemmini.forward(a_load).parent().parent().parent(),
        4,
        ["koo", "koi"],
        perfect=True,
    )

    # print("9: After rearranging dimensions")
    # print(gemmini)

    # Replace to gemmini calls, inline to v2, and hoist all the configurations
    gemmini = set_memory(gemmini, res_alloc, GEMM_ACCUM)
    gemmini = set_memory(gemmini, a_alloc, GEMM_SCRATCH)
    gemmini = set_memory(gemmini, b_alloc, GEMM_SCRATCH)
    print("10: After setting memories")
    print(gemmini)
    print(ld_i8_block_id1)
    tuples = [
        (ld_acc_i32_vector, ld_acc_i32_vector_v2),
        (ld_i8_block_id1, ld_i8_block_id1_v2),
        (ld_i8_block_id2, ld_i8_block_id2_v2),
        (matmul_acc_i8, matmul_acc_i8_v2),
        (st_acc_i8, st_acc_i8_v2),
    ]
    for t in tuples:
        gemmini = simplify(replace_and_inline(gemmini, t))
    print("11: After replacing and inlining")
    print(gemmini)
    gemmini = fuse_all_loops(gemmini, gemmini.body()[0])
    gemmini = simplify(gemmini)
    print("after fusion")
    print(gemmini)
    print(gemmini.c_code_str())


def test_matmul_zz_sample3_optimized():
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
    # method 1
    # gemmini = divide_loop(gemmini, k_loop, 16, ["ko", "ki"], perfect=True)
    # (gemmini, _), a_load, a_alloc = bind_and_lift(
    #     gemmini, gemmini.forward(a_assign).rhs(), max_size=sc_size
    # )
    # gemmini = fission_as_much_as_possible(gemmini, a_load)
    # gemmini = reorder_top(gemmini, gemmini.forward(k_loop))
    # method 2
    gemmini = stage_mem(gemmini, "for i in _:_", "A[0:256, 0:512]", "A_temp")
    gemmini = simplify(gemmini)
    print(gemmini)
    gemmini = divide_dim(gemmini, gemmini.find("A_temp: _"), 1, 16)
    gemmini = divide_dim(gemmini, gemmini.find("A_temp: _"), 0, 16)
    gemmini = rearrange_dim(gemmini, gemmini.find("A_temp: _"), [0, 2, 1, 3])
    print(gemmini)
    i0_loop = gemmini.find_loop("i0")
    i1_loop = gemmini.find_loop("i1")
    gemmini, _ = tile_loops(gemmini, [(i0_loop, 16), (i1_loop, 16)], perfect=True)
    gemmini = divide_loop(gemmini, i1_loop, 4, ["i1oo", "i1oi"], perfect=True)
    print(gemmini)

    # Tile loops for a scratchpad and an accumulator
    # print(gemmini)
    gemmini = reorder_loops(gemmini, "i j")
    gemmini, _ = tile_loops(gemmini, [(j_loop, 16), (i_loop, 16)], perfect=True)
    gemmini = divide_loop(gemmini, j_loop, 4, ["joo", "joi"], perfect=True)
    print(gemmini)
    gemmini = divide_loop(gemmini, k_loop, 16, ["ko", "ki"], perfect=True)
    gemmini = reorder_loops(gemmini, "ji ii")
    # # Bind and lift scratchpad & accumulator memories
    gemmini, _ = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
    # (gemmini, occ_size),  a_load, a_alloc = bind_and_lift(
    #     gemmini, gemmini.forward(a_assign).rhs(), max_size=sc_size
    # )
    # print(occ_size)
    occ_size = 256 * 512
    (gemmini, _), b_load, b_alloc = bind_and_lift(
        gemmini, gemmini.forward(b_assign).rhs(), max_size=sc_size - occ_size
    )
    print("after bind and lift b")
    print(gemmini)
    # # Divide by 4 to use load_blocks
    # gemmini, _ = tile_loops(gemmini, [(k_loop, 4)])
    # gemmini = divide_loop(gemmini, gemmini.find_loop("i0"), 4, ["joo", "joi"], perfect=True)
    # gemmini, [j_imost] = tile_loops(gemmini, [(j_loop, 4)], perfect=True)
    # # # print("before fissioning")
    # # # print(gemmini)
    # Fission all the loops
    gemmini = fission_as_much_as_possible(gemmini, res_load)
    # print("5: After fissioning res_load")
    # print(gemmini)
    gemmini = fission_as_much_as_possible(gemmini, k_loop)
    # print("6: After fissioning k_loop")
    # print(gemmini)
    # gemmini = fission_as_much_as_possible(gemmini, a_load)
    # print("7: After fissioning a_load")
    # print(gemmini)
    gemmini = fission_as_much_as_possible(gemmini, b_load)
    print("8: After fissioning")
    print(gemmini)

    # Fix indexing
    # # gemmini = rearrange_dim(gemmini, res_alloc, [1, 0, 3, 2])
    # gemmini = rearrange_dim(gemmini, a_alloc, [0, 2, 1, 3])
    gemmini = rearrange_dim(gemmini, b_alloc, [2, 0, 3, 1])
    # # gemmini = reorder_loops_from_idx(gemmini, res_load)
    # gemmini = reorder_loops_from_idx(gemmini, a_load)
    gemmini = reorder_loops_from_idx(gemmini, b_load)

    gemmini = reorder_loops_from_idx(gemmini, gemmini.forward(b_assign).rhs())
    gemmini = reorder_loops(gemmini, gemmini.forward(b_assign).parent().parent())
    # print(gemmini)
    # gemmini = remove_redundant_loops(gemmini, a_load, num=2) # original 2: check for final zizag mapping (fusion)
    gemmini = remove_redundant_loops(gemmini, b_load, num=2)
    # print("9: After rearranging dimensions")
    # print(gemmini)

    # Replace to gemmini calls, inline to v2, and hoist all the configurations
    gemmini = set_memory(gemmini, res_alloc, GEMM_ACCUM)
    gemmini = set_memory(gemmini, gemmini.find("A_temp: _"), GEMM_SCRATCH)
    gemmini = set_memory(gemmini, b_alloc, GEMM_SCRATCH)
    print("10: After setting memories")
    print(gemmini)
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
    # print(gemmini)
    # gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id1(_)"))
    # gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id2(_)"))
    # print("12: After adding guards")
    # print(gemmini)

    # # fuse loops, and unroll
    # gemmini = reorder_top(gemmini, gemmini.find("res: _"))
    # gemmini = reorder_top(gemmini, gemmini.find("B_tmp: _"))
    gemmini = fuse_all_loops(gemmini, gemmini.body()[0])
    print("13: After fusing all loops")
    print(gemmini)
    # gemmini = unroll_all(gemmini, gemmini.find_loop("joi", many=True))
    gemmini = simplify(gemmini)
    print(gemmini)
    print(gemmini.c_code_str())


# test_matmul_zz_sample3()
# test_matmul_zz_sample3_optimized()
test_matmul_exo_sample3()

# test_matmul_zz_sample2()
# test_matmul_zz_sample2_optimized()
# test_matmul_exo_sample2()

# test_matmul_zz_sample1()
# test_matmul_zz_sample1_optimized()
# test_matmul_exo_sample1()

### tests for formulation
# test_matmul_zz_sample3_2()
