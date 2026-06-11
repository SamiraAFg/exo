from __future__ import annotations

from exo.platforms.gemmini import *
from exo.stdlib.scheduling import *
from exo.platforms.gemmini_schedules import *


def matmul_algorithm_bias():
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
        C: i8[N, M] @ DRAM,
    ):

        # Algorithm starts here
        for i in seq(0, N):
            for j in seq(0, M):
                res: i32 @ DRAM
                res = 0.0
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


@proc
def matmul_on_cpu(
    N: size,
    M: size,
    K: size,
    scale: f32,
    act: bool,
    A: i8[N, K] @ DRAM,
    B: i8[K, M] @ DRAM,
    C: i8[N, M] @ DRAM,
):
    for i in seq(0, N):
        for j in seq(0, M):
            res: i32 @ DRAM
            res = 0.0
            for k in seq(0, K):
                a: i8 @ DRAM
                a = A[i, k]

                b: i8 @ DRAM
                b = B[k, j]

                a2: i32
                b2: i32
                a2 = a
                b2 = b
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


def sched_matmul(
    name,
    NN,
    MM,
    KK,
):
    cpu = rename(matmul_on_cpu, f"cpu_{name}")
    cpu = cpu.partial_eval(NN, MM, KK)

    gemmini = rename(cpu, name)
    gemmini = set_memory(gemmini, "res", GEMM_ACCUM)
    gemmini = set_memory(gemmini, "a", GEMM_SCRATCH)
    gemmini = set_memory(gemmini, "b", GEMM_SCRATCH)

    # Tile outer loops
    gemmini = tile_outer_loops(gemmini)

    # Lift res, so that we can fission the inner loop to use gemmini instructions
    gemmini = old_lift_alloc(gemmini, "res : _ #0", n_lifts=2)
    gemmini = old_lift_alloc(gemmini, "res : _ #0", n_lifts=1, mode="col", size=16)

    # fission loops to zero accum code block, main block, and store block and reorder k up
    gemmini = fission_outer_blocks(gemmini)

    # fission the main block to 4x16x16 blocks, so that we can use gemmini instr
    gemmini = fission_inner_blocks(gemmini)

    # replace to gemmini calls
    gemmini = replace_gemmini_calls(gemmini)

    # inline and lift config
    gemmini = inline_lift_config(gemmini)

    return cpu, gemmini


# Best for 512x512x512
def schedule_matmul_512x512x512():
    NN = 512
    MM = 512
    KK = 512

    cpu, gemmini = sched_matmul("matmul_512x512x512", NN, MM, KK)

    # Real optimization
    # tile
    gemmini = matmul_tile(gemmini)

    gemmini = old_lift_alloc(gemmini, "res : _", n_lifts=1)
    gemmini = old_lift_alloc(gemmini, "a : _", n_lifts=4)
    gemmini = old_lift_alloc(gemmini, "b : _", n_lifts=3)

    for (s, n) in [("a : i8", 1), ("b : i8", 2), ("res : _", 4)]:
        gemmini = old_lift_alloc(gemmini, s, n_lifts=n, keep_dims=False)

    gemmini = simplify(gemmini)

    def do_fission(pattern, n):
        nonlocal gemmini
        gemmini = autofission(gemmini, gemmini.find(pattern).after(), n_lifts=n)

    do_fission("for j_in_o in _:_", 5)
    do_fission("do_ld_i8_block_id1(_)", 6)
    do_fission("for k in _:_", 6)
    gemmini = add_loop(gemmini, "do_ld_i8_block_id1(_)", "ji", 4, guard=True)
    gemmini = add_loop(gemmini, "if ji == 0: _", "jo", 2, guard=True)
    gemmini = add_loop(gemmini, "do_ld_i8_block_id2(_)", "i", 8, guard=True)
    gemmini = add_loop(gemmini, "if i == 0: _", "io", 2, guard=True)
    # Fuse_loop cleanup
    gemmini = add_loop(gemmini, "for jo in _:_ #1", "ioo", 2)
    gemmini = add_loop(gemmini, "for ji in _:_ #0", "ioo", 2)
    gemmini = fuse(gemmini, "for ioo in _:_ #0", "for ioo in _:_ #1")
    gemmini = fuse(gemmini, "for ioo in _:_ #0", "for ioo in _:_ #1")
    gemmini = fuse(
        gemmini, "for ioo in _:_ #0", "for ioo in _:_ #1", unsafe_disable_check=True
    )
    gemmini = add_loop(gemmini, "for ji in _:_ #0", "jo", 2)
    gemmini = old_reorder(gemmini, "ji jo")
    gemmini = old_reorder(gemmini, "ko jo")
    gemmini = old_reorder(gemmini, "i jo")
    gemmini = old_reorder(gemmini, "io jo")
    gemmini = fuse(gemmini, "for jo in _:_ #0", "for jo in _:_ #1")
    gemmini = fuse(gemmini, "for jo in _:_ #0", "for jo in _:_ #1")
    gemmini = fuse(
        gemmini, "for jo in _:_ #0", "for jo in _:_ #1", unsafe_disable_check=True
    )
    gemmini = old_reorder(gemmini, "i io")
    gemmini = old_reorder(gemmini, "k io")
    gemmini = old_reorder(gemmini, "ko io")
    gemmini = old_reorder(gemmini, "ji io")
    gemmini = add_loop(gemmini, "for ji in _:_ #0", "io", 2)
    gemmini = fuse(gemmini, "for io in _:_ #0", "for io in _:_ #1")
    gemmini = fuse(gemmini, "for io in _:_ #0", "for io in _:_ #1")
    gemmini = fuse(
        gemmini, "for io in _:_ #0", "for io in _:_ #1", unsafe_disable_check=True
    )
    gemmini = old_reorder(gemmini, "k i")
    gemmini = old_reorder(gemmini, "ko i")
    gemmini = old_reorder(gemmini, "ji i")
    gemmini = add_loop(gemmini, "for ji in _:_ #0", "i", 8)
    gemmini = fuse(gemmini, "for i in _:_ #0", "for i in _:_ #1")
    gemmini = fuse(gemmini, "for i in _:_ #0", "for i in _:_ #1")
    gemmini = fuse(
        gemmini, "for i in _:_ #0", "for i in _:_ #1", unsafe_disable_check=True
    )
    gemmini = old_reorder(gemmini, "ko ji")
    gemmini = fuse(gemmini, "for ji in _:_ #0", "for ji in _:_ #1")
    gemmini = fuse(gemmini, "for ji in _:_ #0", "for ji in _:_ #1")
    gemmini = fuse(gemmini, "for ji in _:_ #0", "for ji in _:_ #1")
    gemmini = fuse(gemmini, "for ko in _:_ #0", "for ko in _:_ #1")
    gemmini = fuse(gemmini, "for ko in _:_ #0", "for ko in _:_ #1")

    gemmini = fuse(gemmini, "for k in _:_ #0", "for k in _:_ #1")
    gemmini = old_unroll(gemmini, "j_in_o")
    gemmini = old_unroll(gemmini, "k")
    gemmini = simplify(gemmini)

    return cpu, gemmini


def schedule_matmul_4():
    NN = 12544
    MM = 256
    KK = 64

    cpu, gemmini = sched_matmul("matmul_4", NN, MM, KK)

    # Real optimization
    gemmini = old_unroll(gemmini, "ko")
    gemmini = old_lift_alloc(gemmini, "res:_")
    gemmini = simplify(gemmini)

    gemmini = divide_loop(gemmini, "i", 196, ["io", "i"], perfect=True)
    gemmini = old_lift_alloc(gemmini, "a : _", n_lifts=2)
    gemmini = old_lift_alloc(gemmini, "b : _", n_lifts=2)

    # tile
    gemmini = old_lift_alloc(gemmini, "a : i8", n_lifts=1, keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "b : i8", n_lifts=1, keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "res : _", n_lifts=2, keep_dims=False)

    # Previously add_guard
    gemmini = simplify(gemmini)

    def do_fission(pattern, n):
        nonlocal gemmini
        gemmini = autofission(gemmini, gemmini.find(pattern).after(), n_lifts=n)

    do_fission("for j_in_o in _:_", 5)
    do_fission("do_ld_i8_block_id1(_)", 6)
    do_fission("for k in _:_", 6)
    gemmini = add_loop(gemmini, "do_ld_i8_block_id1(_)", "j", 4, guard=True)
    gemmini = add_loop(gemmini, "do_ld_i8_block_id2(_)", "i", 196, guard=True)
    gemmini = add_loop(gemmini, "if i == 0: _", "io", 4, guard=True)
    # Fuse_loop cleanup
    gemmini = old_reorder(gemmini, "i io")
    gemmini = old_reorder(gemmini, "k io")
    gemmini = old_reorder(gemmini, "j io")
    gemmini = add_loop(gemmini, "for j in _:_ #0", "io", 4)
    gemmini = fuse(gemmini, "for io in _:_ #0", "for io in _:_ #1")
    gemmini = fuse(gemmini, "for io in _:_ #0", "for io in _:_ #1")
    gemmini = fuse(
        gemmini, "for io in _:_ #0", "for io in _:_ #1", unsafe_disable_check=True
    )
    gemmini = add_loop(gemmini, "for j_in_o in _:_ #0", "i", 196)
    gemmini = old_reorder(gemmini, "k i")
    gemmini = old_reorder(gemmini, "j i")
    gemmini = fuse(gemmini, "for i in _:_ #0", "for i in _:_ #1")
    gemmini = fuse(gemmini, "for i in _:_ #0", "for i in _:_ #1")
    gemmini = fuse(
        gemmini, "for i in _:_ #0", "for i in _:_ #1", unsafe_disable_check=True
    )
    gemmini = fuse(gemmini, "for j in _:_ #0", "for j in _:_ #1")
    gemmini = fuse(gemmini, "for j in _:_ #0", "for j in _:_ #1")
    gemmini = fuse(gemmini, "for j in _:_ #0", "for j in _:_ #1")

    gemmini = old_unroll(gemmini, "j_in_o")
    gemmini = old_unroll(gemmini, "k")

    return cpu, gemmini


def schedule_matmul_6():
    NN = 12544
    MM = 64
    KK = 256

    cpu, gemmini = sched_matmul("matmul_6", NN, MM, KK)

    # Real optimization
    gemmini = old_unroll(gemmini, "j")
    gemmini = divide_loop(gemmini, "i", 8, ["io", "i"], perfect=True)
    gemmini = old_lift_alloc(gemmini, "res:_")
    gemmini = old_lift_alloc(gemmini, "a : _")
    gemmini = old_lift_alloc(gemmini, "b : _")
    gemmini = simplify(gemmini)

    # tile
    gemmini = old_lift_alloc(gemmini, "res:_", keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "a : _", keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "b : _", keep_dims=False)

    gemmini = simplify(gemmini)

    # Previously add_guard
    def do_fission(pattern, n):
        nonlocal gemmini
        gemmini = autofission(gemmini, gemmini.find(pattern).after(), n_lifts=n)

    do_fission("for j_in_o in _:_", 2)
    do_fission("do_ld_i8_block_id1(_)", 3)
    do_fission("for k in _:_", 3)
    gemmini = add_loop(gemmini, "do_ld_i8_block_id2(_)", "i", 8, guard=True)
    gemmini = add_loop(gemmini, "if i == 0: _", "io", 98, guard=True)
    # Fuse_loop cleanup
    gemmini = old_reorder(gemmini, "i io")
    gemmini = old_reorder(gemmini, "k io")
    gemmini = old_reorder(gemmini, "ko io")
    gemmini = add_loop(gemmini, "for i in _:_ #0", "io", 98)
    gemmini = fuse(gemmini, "for io in _:_ #0", "for io in _:_ #1")
    gemmini = fuse(gemmini, "for io in _:_ #0", "for io in _:_ #1")
    gemmini = fuse(
        gemmini, "for io in _:_ #0", "for io in _:_ #1", unsafe_disable_check=True
    )
    gemmini = old_reorder(gemmini, "k i")
    gemmini = old_reorder(gemmini, "ko i")
    gemmini = fuse(gemmini, "for i in _:_ #0", "for i in _:_ #1")
    gemmini = fuse(gemmini, "for i in _:_ #0", "for i in _:_ #1")
    gemmini = fuse(gemmini, "for i in _:_ #0", "for i in _:_ #1")
    gemmini = fuse(gemmini, "for ko in _:_ #0", "for ko in _:_ #1")
    gemmini = fuse(gemmini, "for ko in _:_ #0", "for ko in _:_ #1")

    gemmini = old_unroll(gemmini, "j_in_o")
    gemmini = old_unroll(gemmini, "k")

    return cpu, gemmini


def schedule_matmul_14():
    NN = 3136
    MM = 512
    KK = 128

    cpu, gemmini = sched_matmul("matmul_14", NN, MM, KK)

    # Real optimization
    gemmini = divide_loop(gemmini, "i", 49, ["io", "i"], perfect=True)
    gemmini = old_lift_alloc(gemmini, "res:_")
    gemmini = old_lift_alloc(gemmini, "a : _", n_lifts=2)
    gemmini = old_lift_alloc(gemmini, "b : _", n_lifts=2)
    gemmini = simplify(gemmini)

    # tile
    gemmini = old_lift_alloc(gemmini, "res:_", n_lifts=2, keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "a : _", keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "b : _", keep_dims=False)

    gemmini = simplify(gemmini)

    # Previously add_guard
    def do_fission(pattern, n):
        nonlocal gemmini
        gemmini = autofission(gemmini, gemmini.find(pattern).after(), n_lifts=n)

    do_fission("for j_in_o in _:_", 3)
    do_fission("do_ld_i8_block_id1(_)", 4)
    do_fission("for k in _:_", 4)
    gemmini = add_loop(gemmini, "do_ld_i8_block_id1(_)", "j", 8, guard=True)
    gemmini = add_loop(gemmini, "do_ld_i8_block_id2(_)", "i", 49, guard=True)
    gemmini = add_loop(gemmini, "if i == 0: _", "io", 4, guard=True)
    # Fuse_loop cleanup
    gemmini = add_loop(gemmini, "for j in _:_ #0", "io", 4)
    gemmini = old_reorder(gemmini, "i io")
    gemmini = old_reorder(gemmini, "k io")
    gemmini = old_reorder(gemmini, "ko io")
    gemmini = old_reorder(gemmini, "j io")
    gemmini = fuse(gemmini, "for io in _:_ #0", "for io in _:_ #1")
    gemmini = fuse(gemmini, "for io in _:_ #0", "for io in _:_ #1")
    gemmini = fuse(
        gemmini, "for io in _:_ #0", "for io in _:_ #1", unsafe_disable_check=True
    )
    gemmini = add_loop(gemmini, "for j in _:_ #0", "i", 49)
    gemmini = old_reorder(gemmini, "k i")
    gemmini = old_reorder(gemmini, "ko i")
    gemmini = old_reorder(gemmini, "j i")
    gemmini = fuse(gemmini, "for i in _:_ #0", "for i in _:_ #1")
    gemmini = fuse(gemmini, "for i in _:_ #0", "for i in _:_ #1")
    gemmini = fuse(
        gemmini, "for i in _:_ #0", "for i in _:_ #1", unsafe_disable_check=True
    )
    gemmini = old_reorder(gemmini, "ko j")
    gemmini = fuse(gemmini, "for j in _:_ #0", "for j in _:_ #1")
    gemmini = fuse(gemmini, "for j in _:_ #0", "for j in _:_ #1")
    gemmini = fuse(gemmini, "for j in _:_ #0", "for j in _:_ #1")
    gemmini = fuse(gemmini, "for ko in _:_ #0", "for ko in _:_ #1")
    gemmini = fuse(gemmini, "for ko in _:_ #0", "for ko in _:_ #1")

    gemmini = old_unroll(gemmini, "j_in_o")
    gemmini = old_unroll(gemmini, "k")

    return cpu, gemmini


def schedule_matmul_16():
    NN = 3136
    MM = 128
    KK = 512

    cpu, gemmini = sched_matmul("matmul_16", NN, MM, KK)

    # Real optimization
    gemmini = divide_loop(gemmini, "i", 14, ["io", "i"], perfect=True)
    gemmini = old_lift_alloc(gemmini, "res:_")
    gemmini = old_lift_alloc(gemmini, "a : _", n_lifts=2)
    gemmini = old_lift_alloc(gemmini, "b : _", n_lifts=2)
    gemmini = simplify(gemmini)

    # tile
    gemmini = old_lift_alloc(gemmini, "res:_", n_lifts=2, keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "a : _", keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "b : _", keep_dims=False)

    gemmini = simplify(gemmini)

    # Previously add_guard
    def do_fission(pattern, n):
        nonlocal gemmini
        gemmini = autofission(gemmini, gemmini.find(pattern).after(), n_lifts=n)

    do_fission("for j_in_o in _:_", 3)
    do_fission("do_ld_i8_block_id1(_)", 4)
    do_fission("for k in _:_", 4)
    gemmini = add_loop(gemmini, "do_ld_i8_block_id1(_)", "j", 2, guard=True)
    gemmini = add_loop(gemmini, "do_ld_i8_block_id2(_)", "i", 14, guard=True)
    gemmini = add_loop(gemmini, "if i == 0: _", "io", 14, guard=True)
    # Fuse_loop cleanup
    gemmini = add_loop(gemmini, "for j in _:_ #0", "io", 14)
    gemmini = old_reorder(gemmini, "i io")
    gemmini = old_reorder(gemmini, "k io")
    gemmini = old_reorder(gemmini, "ko io")
    gemmini = old_reorder(gemmini, "j io")
    gemmini = fuse(gemmini, "for io in _:_ #0", "for io in _:_ #1")
    gemmini = fuse(gemmini, "for io in _:_ #0", "for io in _:_ #1")
    gemmini = fuse(
        gemmini, "for io in _:_ #0", "for io in _:_ #1", unsafe_disable_check=True
    )
    gemmini = add_loop(gemmini, "for j in _:_ #0", "i", 14)
    gemmini = old_reorder(gemmini, "k i")
    gemmini = old_reorder(gemmini, "ko i")
    gemmini = old_reorder(gemmini, "j i")
    gemmini = fuse(gemmini, "for i in _:_ #0", "for i in _:_ #1")
    gemmini = fuse(gemmini, "for i in _:_ #0", "for i in _:_ #1")
    gemmini = fuse(
        gemmini, "for i in _:_ #0", "for i in _:_ #1", unsafe_disable_check=True
    )
    gemmini = old_reorder(gemmini, "ko j")
    gemmini = fuse(gemmini, "for j in _:_ #0", "for j in _:_ #1")
    gemmini = fuse(gemmini, "for j in _:_ #0", "for j in _:_ #1")
    gemmini = fuse(gemmini, "for j in _:_ #0", "for j in _:_ #1")
    gemmini = fuse(gemmini, "for ko in _:_ #0", "for ko in _:_ #1")
    gemmini = fuse(gemmini, "for ko in _:_ #0", "for ko in _:_ #1")

    gemmini = old_unroll(gemmini, "j_in_o")
    gemmini = old_unroll(gemmini, "k")

    return cpu, gemmini


def schedule_matmul_27():
    NN = 784
    MM = 1024
    KK = 256

    cpu, gemmini = sched_matmul("matmul_27", NN, MM, KK)

    # Real optimization
    gemmini = divide_loop(gemmini, "i", 7, ["io", "i"], perfect=True)
    gemmini = divide_loop(gemmini, "j", 8, ["jo", "j"], perfect=True)
    gemmini = old_reorder(gemmini, "i jo")
    gemmini = old_lift_alloc(gemmini, "res:_")
    gemmini = old_lift_alloc(gemmini, "a : _", n_lifts=2)
    gemmini = old_lift_alloc(gemmini, "b : _", n_lifts=2)
    gemmini = simplify(gemmini)

    # tile
    gemmini = old_lift_alloc(gemmini, "res:_", n_lifts=3, keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "a : _", n_lifts=2, keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "b : _", n_lifts=2, keep_dims=False)

    gemmini = simplify(gemmini)

    # Previously add_guard
    def do_fission(pattern, n):
        nonlocal gemmini
        gemmini = autofission(gemmini, gemmini.find(pattern).after(), n_lifts=n)

    do_fission("for j_in_o in _:_", 4)
    do_fission("do_ld_i8_block_id1(_)", 5)
    do_fission("for k in _:_", 5)
    gemmini = add_loop(gemmini, "do_ld_i8_block_id1(_)", "j", 8, guard=True)
    gemmini = add_loop(gemmini, "if j == 0: _", "jo", 2, guard=True)
    gemmini = add_loop(gemmini, "do_ld_i8_block_id2(_)", "i", 7, guard=True)
    # Fuse_loop cleanup
    gemmini = add_loop(gemmini, "for jo in _:_ #1", "io", 7)
    gemmini = add_loop(gemmini, "for j in _:_ #0", "io", 7)
    gemmini = fuse(gemmini, "for io in _:_ #0", "for io in _:_ #1")
    gemmini = fuse(gemmini, "for io in _:_ #0", "for io in _:_ #1")
    gemmini = fuse(
        gemmini, "for io in _:_ #0", "for io in _:_ #1", unsafe_disable_check=True
    )
    gemmini = add_loop(gemmini, "for j in _:_ #0", "jo", 2)
    gemmini = old_reorder(gemmini, "j jo")
    gemmini = old_reorder(gemmini, "ko jo")
    gemmini = old_reorder(gemmini, "i jo")
    gemmini = fuse(gemmini, "for jo in _:_ #0", "for jo in _:_ #1")
    gemmini = fuse(gemmini, "for jo in _:_ #0", "for jo in _:_ #1")
    gemmini = fuse(
        gemmini, "for jo in _:_ #0", "for jo in _:_ #1", unsafe_disable_check=True
    )
    gemmini = old_reorder(gemmini, "k i")
    gemmini = old_reorder(gemmini, "ko i")
    gemmini = old_reorder(gemmini, "j i")
    gemmini = add_loop(gemmini, "for j in _:_ #0", "i", 7)
    gemmini = fuse(gemmini, "for i in _:_ #0", "for i in _:_ #1")
    gemmini = fuse(gemmini, "for i in _:_ #0", "for i in _:_ #1")
    gemmini = fuse(
        gemmini, "for i in _:_ #0", "for i in _:_ #1", unsafe_disable_check=True
    )
    gemmini = old_reorder(gemmini, "ko j")
    gemmini = fuse(gemmini, "for j in _:_ #0", "for j in _:_ #1")
    gemmini = fuse(gemmini, "for j in _:_ #0", "for j in _:_ #1")
    gemmini = fuse(gemmini, "for j in _:_ #0", "for j in _:_ #1")
    gemmini = fuse(gemmini, "for ko in _:_ #0", "for ko in _:_ #1")
    gemmini = fuse(gemmini, "for ko in _:_ #0", "for ko in _:_ #1")

    gemmini = old_unroll(gemmini, "j_in_o")
    gemmini = old_unroll(gemmini, "k")

    return cpu, gemmini


ld_i8_block_id1 = reorder_loops(ld_i8_block_id1, "i j")
ld_i8_block_id2 = reorder_loops(ld_i8_block_id2, "i j")


def schedule_matmul_XxXx512_new():
    KK = 512
    # NN = 512
    # MM = 512

    cpu = rename(
        matmul_algorithm(), "cpu_matmul_XxXx512_new"
    )  # Rename "matmul" to "matmul_on_cpu"
    cpu = cpu.partial_eval(K=KK)
    # cpu = cpu.partial_eval(N=NN, M=MM, k=KK)
    cpu = cpu.add_assertion("N % 256 == 0")
    cpu = cpu.add_assertion("M % 256 == 0")

    # Rename the procedure to "matmul_on_gemmini"
    gemmini = rename(cpu, "matmul_XxXx512_new")

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
    res_load = gemmini.find("res = 0.0")
    a_assign = gemmini.find("a2 = A[_]")
    b_assign = gemmini.find("b2 = B[_]")
    res_alloc = res_load.prev()

    # Schedule starts here!!

    # Tile loops for a scratchpad and an accumulator
    gemmini, _ = tile_loops(gemmini, [(i_loop, 16), (j_loop, 16)], perfect=True)
    gemmini, [_, j_outer] = tile_loops(
        gemmini, [(i_loop, 16), (j_loop, 16)], perfect=True
    )
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
    gemmini, [j_imost] = tile_loops(gemmini, [(j_outer, 4)], perfect=True)

    # Fission all the loops
    gemmini = fission_as_much_as_possible(gemmini, res_load)
    gemmini = fission_as_much_as_possible(gemmini, k_loop)
    gemmini = fission_as_much_as_possible(gemmini, a_load)
    gemmini = fission_as_much_as_possible(gemmini, b_load)

    # Fix indexing
    gemmini = rearrange_dim(gemmini, a_alloc, [0, 2, 1, 3])
    gemmini = rearrange_dim(gemmini, b_alloc, [2, 0, 3, 1])
    gemmini = reorder_loops_from_idx(gemmini, a_load)
    gemmini = reorder_loops_from_idx(gemmini, b_load)
    gemmini = reorder_loops_from_idx(gemmini, gemmini.forward(a_assign).rhs())
    gemmini = remove_redundant_loops(gemmini, a_load, num=2)
    gemmini = remove_redundant_loops(gemmini, b_load, num=1)

    # Replace to gemmini calls, inline to v2, and hoist all the configurations
    gemmini = set_memory(gemmini, res_alloc, GEMM_ACCUM)
    gemmini = set_memory(gemmini, a_alloc, GEMM_SCRATCH)
    gemmini = set_memory(gemmini, b_alloc, GEMM_SCRATCH)
    tuples = [
        (zero_acc_i32, zero_acc_i32_v2),
        (ld_i8_block_id1, ld_i8_block_id1_v2),
        (ld_i8_block_id2, ld_i8_block_id2_v2),
        (matmul_acc_i8, matmul_acc_i8_v2),
        (st_acc_i8, st_acc_i8_v2),
    ]
    for t in tuples:
        gemmini = simplify(replace_and_inline(gemmini, t))

    # Add a guard to redundant loads
    gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id1(_)"))
    gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id2(_)"))

    # fuse loops, and unroll
    gemmini = fuse_all_loops(gemmini, gemmini.body()[0])
    gemmini = unroll_all(gemmini, gemmini.find_loop(j_imost.name(), many=True))

    # Schedule ends here!!
    # 29 lines excluding comments and newlines

    gemmini = simplify(gemmini)

    # print("")
    # print("============= THIS IS THE SCHEDULED MATMUL ===============")
    # print(gemmini)
    # print("============= THIS IS THE SCHEDULED MATMUL ===============")
    # print("")
    return cpu, gemmini


def schedule_matmul_XxXx512_new_bias():
    KK = 512
    # NN = 512
    # MM = 512

    cpu = rename(
        matmul_algorithm_bias(), "cpu_matmul_XxXx512_new_bias"
    )  # Rename "matmul" to "matmul_on_cpu"
    cpu = cpu.partial_eval(K=KK)
    # cpu = cpu.partial_eval(N=NN, M=MM, k=KK)
    cpu = cpu.add_assertion("N % 256 == 0")
    cpu = cpu.add_assertion("M % 256 == 0")

    # Rename the procedure to "matmul_on_gemmini"
    gemmini = rename(cpu, "matmul_XxXx512_new_bias")

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
    gemmini, [_, j_outer] = tile_loops(
        gemmini, [(i_loop, 16), (j_loop, 16)], perfect=True
    )
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
    gemmini, [j_imost] = tile_loops(gemmini, [(j_outer, 4)], perfect=True)

    # Fission all the loops
    gemmini = fission_as_much_as_possible(gemmini, res_load)
    gemmini = fission_as_much_as_possible(gemmini, k_loop)
    gemmini = fission_as_much_as_possible(gemmini, a_load)
    gemmini = fission_as_much_as_possible(gemmini, b_load)

    # Fix indexing
    gemmini = rearrange_dim(gemmini, a_alloc, [0, 2, 1, 3])
    gemmini = rearrange_dim(gemmini, b_alloc, [2, 0, 3, 1])
    gemmini = reorder_loops_from_idx(gemmini, a_load)
    gemmini = reorder_loops_from_idx(gemmini, b_load)
    gemmini = reorder_loops_from_idx(gemmini, gemmini.forward(a_assign).rhs())
    gemmini = remove_redundant_loops(gemmini, a_load, num=2)
    gemmini = remove_redundant_loops(gemmini, b_load, num=1)

    # Replace to gemmini calls, inline to v2, and hoist all the configurations
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

    # Add a guard to redundant loads
    gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id1(_)"))
    gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id2(_)"))

    # fuse loops, and unroll
    gemmini = fuse_all_loops(gemmini, gemmini.body()[0])
    gemmini = unroll_all(gemmini, gemmini.find_loop(j_imost.name(), many=True))

    # Schedule ends here!!
    # 29 lines excluding comments and newlines

    gemmini = simplify(gemmini)

    # print("")
    # print("============= THIS IS THE SCHEDULED MATMUL ===============")
    # print(gemmini)
    # print("============= THIS IS THE SCHEDULED MATMUL ===============")
    # print("")
    return cpu, gemmini


def schedule_matmul_128_bias():
    KK = 128
    NN = 128
    MM = 128

    cpu = rename(
        matmul_algorithm_bias(), "cpu_matmul_128_bias"
    )  # Rename "matmul" to "matmul_on_cpu"
    # cpu = cpu.partial_eval(K=KK)
    cpu = cpu.partial_eval(N=NN, M=MM, K=KK)
    # cpu = cpu.add_assertion("N % 256 == 0")
    # cpu = cpu.add_assertion("M % 256 == 0")

    # Rename the procedure to "matmul_on_gemmini"
    gemmini = rename(cpu, "matmul_128_bias")

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

    # print("")
    # print("============= THIS IS THE SCHEDULED MATMUL ===============")
    # print(gemmini)
    # print("============= THIS IS THE SCHEDULED MATMUL ===============")
    # print("")
    return cpu, gemmini


def schedule_matmul_128():
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
    res_load = gemmini.find("res = 0.0")
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
        (zero_acc_i32, zero_acc_i32_v2),
        # (ld_acc_i32_vector, ld_acc_i32_vector_v2),
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

    # print("")
    # print("============= THIS IS THE SCHEDULED MATMUL ===============")
    # print(gemmini)
    # print("============= THIS IS THE SCHEDULED MATMUL ===============")
    # print("")
    return cpu, gemmini


def test_matmul_all(NN, MM, KK):
    # KK = 512
    assert NN % 64 == 0, "N must be a multiple of 64"
    assert MM % 64 == 0, "M must be a multiple of 64"
    assert KK % 64 == 0, "K must be a multiple of 64"
    cpu = rename(
        matmul_algorithm_bias(), f"cpu_exo2_matmul_{NN}x{MM}x{KK}"
    )  # Rename "matmul" to "matmul_on_cpu"
    cpu = cpu.partial_eval(K=KK, N=NN, M=MM)

    # cpu = cpu.add_assertion("N % 256 == 0")
    # cpu = cpu.add_assertion("M % 256 == 0")

    # Rename the procedure to "matmul_on_gemmini"
    gemmini = rename(cpu, f"exo2_matmul_{NN}x{MM}x{KK}")

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

    # print("")
    # print("============= THIS IS THE SCHEDULED MATMUL ===============")
    # print(gemmini)
    # print("============= THIS IS THE SCHEDULED MATMUL ===============")
    # print("")
    return cpu, gemmini


def test_matmul_all_explore_1(NN, MM, KK):
    # KK = 512
    assert NN % 64 == 0, "N must be a multiple of 64"
    assert MM % 64 == 0, "M must be a multiple of 64"
    assert KK % 64 == 0, "K must be a multiple of 64"
    cpu = rename(
        matmul_algorithm_bias(), f"cpu_exo2_matmul_explore1_{NN}x{MM}x{KK}"
    )  # Rename "matmul" to "matmul_on_cpu"
    cpu = cpu.partial_eval(K=KK, N=NN, M=MM)

    # cpu = cpu.add_assertion("N % 256 == 0")
    # cpu = cpu.add_assertion("M % 256 == 0")

    # Rename the procedure to "matmul_on_gemmini"
    gemmini = rename(cpu, f"exo2_matmul_explore1_{NN}x{MM}x{KK}")

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

    # print("")
    # print("============= THIS IS THE SCHEDULED MATMUL ===============")
    # print(gemmini)
    # print("============= THIS IS THE SCHEDULED MATMUL ===============")
    # print("")
    return cpu, gemmini


def test_matmul():
    KK = 512
    N = 256
    M = 1024
    cpu = rename(
        matmul_algorithm_bias(), "cpu_exo2_matmul_zz"
    )  # Rename "matmul" to "matmul_on_cpu"
    cpu = cpu.partial_eval(K=KK)
    cpu = cpu.add_assertion("N % 256 == 0")
    cpu = cpu.add_assertion("M % 256 == 0")

    # Rename the procedure to "matmul_on_gemmini"
    gemmini = rename(cpu, "exo2_matmul_zz")

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
    return cpu, gemmini


cpu_matmul_512x512x512, matmul_512x512x512 = schedule_matmul_512x512x512()
cpu_matmul_4, matmul_4 = schedule_matmul_4()
cpu_matmul_6, matmul_6 = schedule_matmul_6()
cpu_matmul_14, matmul_14 = schedule_matmul_14()
cpu_matmul_16, matmul_16 = schedule_matmul_16()
cpu_matmul_27, matmul_27 = schedule_matmul_27()


# First Tests with EXO2
cpu_exo2_matmul_512x512x512, exo2_matmul_512x512x512 = test_matmul_all(512, 512, 512)
cpu_exo2_matmul_256x256x256, exo2_matmul_256x256x256 = test_matmul_all(256, 256, 256)
cpu_exo2_matmul_128x128x128, exo2_matmul_128x128x128 = test_matmul_all(128, 128, 128)
cpu_exo2_matmul_64x64x64, exo2_matmul_64x64x64 = test_matmul_all(64, 64, 64)

# Exploration tests
# cpu_exo2_matmul_explore1_128x128x128, exo2_matmul_explore1_128x128x128 = test_matmul_all_explore_1(128, 128, 128)

##ZigZag tests
cpu_exo2_matmul_zz, exo2_matmul_zz = test_matmul()

__all__ = [
    # "cpu_matmul_128_bias",
    # "matmul_128_bias",
    # "cpu_matmul_128",
    # "matmul_128",
    # "cpu_matmul_XxXx512_new_bias",
    # "matmul_XxXx512_new_bias",
    # "cpu_matmul_XxXx512_new",
    # "matmul_XxXx512_new",
    # "cpu_exo2_matmul_explore1_128x128x128", "exo2_matmul_explore1_128x128x128",
    "cpu_exo2_matmul_zz",
    "exo2_matmul_zz",
    "cpu_exo2_matmul_512x512x512",
    "exo2_matmul_512x512x512",
    "cpu_exo2_matmul_256x256x256",
    "exo2_matmul_256x256x256",
    "cpu_exo2_matmul_128x128x128",
    "exo2_matmul_128x128x128",
    "cpu_exo2_matmul_64x64x64",
    "exo2_matmul_64x64x64",
    "cpu_matmul_512x512x512",
    "matmul_512x512x512",
    "cpu_matmul_4",
    "matmul_4",
    "cpu_matmul_6",
    "matmul_6",
    "cpu_matmul_14",
    "matmul_14",
    "cpu_matmul_16",
    "matmul_16",
    "cpu_matmul_27",
    "matmul_27",
]
