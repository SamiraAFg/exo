from __future__ import annotations

from exo.platforms.gemmini import *

# import exo.API_cursors as pc
# from exo.libs.memories import GEMM_SCRATCH, GEMM_ACCUM
# from exo import proc, instr, DRAM, config, ExoType
# from exo.stdlib.scheduling import *
# from exo.stdlib.stdlib import *
# from exo.stdlib.inspection import *
# import os
# import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gemmini_schedules import *


@proc
def conv_on_cpu_stride_1(
    batch_size: size,
    out_dim: size,
    out_channel: size,
    kernel_dim: size,
    in_channel: size,
    in_dim: size,
    output: i8[batch_size, out_dim, out_dim, out_channel],
    bias: i32[1, out_channel],
    inp: i8[batch_size, in_dim, in_dim, in_channel],
    weights: i8[kernel_dim, kernel_dim, in_channel, out_channel],
    act: bool,
    scale: f32,
):
    assert out_dim == in_dim - kernel_dim + 1

    for b in seq(0, batch_size):
        for orow in seq(0, out_dim):
            for ocol in seq(0, out_dim):
                for j in seq(0, out_channel):

                    res: i32
                    res = bias[0, j]
                    for krow in seq(0, kernel_dim):
                        for kcol in seq(0, kernel_dim):
                            for kch in seq(0, in_channel):
                                w_s: i8 @ DRAM
                                w_s = weights[krow, kcol, kch, j]

                                i_s: i8 @ DRAM
                                i_s = inp[b, orow + krow, ocol + kcol, kch]

                                a2: i32
                                b2: i32
                                a2 = i_s
                                b2 = w_s

                                res += a2 * b2

                    src_tmp: i32
                    src_tmp = res
                    tmp_res1: f32
                    acc_scale(src_tmp, tmp_res1, scale)
                    tmp_res2: i8
                    clamp(tmp_res1, tmp_res2)
                    if act == True:
                        tmp_res2 = relu(tmp_res2)

                    output[b, orow, ocol, j] = tmp_res2


def my_schedule_conv_example():
    batch_size = 4
    out_channel = 64
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

    accum_size = 16 * 1024
    sc_size = 256 * 1024

    # Grab cursors
    b_loop = conv.find_loop("b")
    krow_loop = conv.find_loop("krow")
    kcol_loop = conv.find_loop("kcol")
    j_loop = conv.find_loop("j")
    kch_loop = conv.find_loop("kch")
    orow_loop = conv.find_loop("orow")
    res_load = conv.find("res = _")
    ocol_loop = conv.find_loop("ocol")
    wgt_assign = conv.find("w_s = weights[_]")
    in_assign = conv.find("i_s = inp[_]")
    res_alloc = res_load.prev()
    # wgt_alloc = wgt_assign.prev()
    # in_alloc = in_assign.prev()

    print("original conv:")
    print(conv)
    # Tile loops for a scratchpad and an accumulator
    # conv, _ = tile_loops(conv, [(och_loop, 16), (kch_loop, 16)], perfect=True)
    # conv, _ = tile_loops(conv, [(och_loop, 16)])
    # conv, _ = tile_loops(conv, [(kch_loop, 16)], perfect=True)
    # conv, _ = tile_loops(conv, [(orow_loop, 16), (ocol_loop, 16)], perfect=True)

    # conv, a_load, a_alloc = bind_and_lift(
    #     conv, conv.forward(wgt_assign).rhs(), max_size=sc_size / 2
    # )
    # conv, b_load, b_alloc = bind_and_lift(
    #     conv, conv.forward(in_assign).rhs(), max_size=sc_size / 2
    # )

    # conv = fission_as_much_as_possible(conv, res_load)
    # # conv = fission_as_much_as_possible(conv, k_loop)
    # conv = fission_as_much_as_possible(conv, a_load)
    # conv = fission_as_much_as_possible(conv, b_load)
    conv = simplify(mult_loops(conv, orow_loop, "ii"))
    conv = simplify(mult_loops(conv, b_loop, "i"))
    conv = simplify(mult_loops(conv, kcol_loop, "ki"))
    conv = simplify(mult_loops(conv, krow_loop, "k"))

    # conv, _ = autolift_alloc(conv, res_alloc, max_size=accum_size)
    # conv, _ = autolift_alloc(conv, wgt_alloc, max_size=sc_size / 2)
    # conv, _ = autolift_alloc(conv, in_alloc, max_size=sc_size / 2)

    print("after :")
    print(conv)


def schedule_conv_3():
    batch_size = 4
    out_channel = 64
    kernel_dim = 3
    in_channel = 64
    in_dim = 58
    out_dim = int((in_dim - kernel_dim) / 1 + 1)
    assert out_dim == 56

    cpu = rename(conv_on_cpu_stride_1, "conv_3_cpu")
    cpu = cpu.partial_eval(
        batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim
    )

    conv = rename(cpu, "conv_3")
    print("original conv:")
    print(conv)

    conv = split_fission_dim(conv)
    print("after split_fission_dim:")
    print(conv)

    conv = replace_div_part(conv)
    conv = replace_mod_part(conv)

    conv = set_gemm_memories(conv)

    conv = inline_div_part(conv)
    conv = inline_mod_part(conv)

    # Real optimization
    conv = old_lift_alloc(conv, "w_s : _", n_lifts=2)
    conv = old_fission_after(conv, "for ocol_o in _:_ #0")
    conv = old_reorder(conv, "orow ocol_o")
    conv = old_split(conv, "orow", 28, ["orow_o", "orow_i"], perfect=True)
    conv = expand_dim(conv, "i_s: i8[_]", "30", "krow + orow_i")
    conv = expand_dim(conv, "i_s: i8[_] #1", "30", "krow + orow_i")

    # TODO: We should definitely use repeat for this!
    conv = old_lift_alloc(conv, "i_s : _ #0", n_lifts=5, keep_dims=False)
    conv = old_lift_alloc(conv, "i_s : _ #1", n_lifts=4, keep_dims=False)
    conv = old_lift_alloc(conv, "w_s : _ #0", n_lifts=4, keep_dims=False)
    conv = old_lift_alloc(conv, "w_s : _ #1", n_lifts=3, keep_dims=False)
    conv = old_lift_alloc(conv, "res : _ #0", n_lifts=4, keep_dims=False)
    conv = old_lift_alloc(conv, "res : _ #1", n_lifts=3, keep_dims=False)

    conv = old_fission_after(conv, "for kch_o in _:_ #0", n_lifts=6)
    conv = old_fission_after(conv, "for kch_o in _:_ #2", n_lifts=5)
    conv = old_fission_after(conv, "for och_o in _:_ #3")
    conv = add_loop(conv, "for kch_o in _:_ #0", "orow_i", 28, guard=True)
    conv = add_loop(conv, "if orow_i == 0:_", "orow_o", 2, guard=True)
    conv = add_loop(conv, "if orow_o == 0:_", "b", 4, guard=True)
    conv = add_loop(conv, "if b == 0:_", "ocol_o", 3, guard=True)
    conv = add_loop(conv, "for kch_o in _:_ #2", "orow_i", 28, guard=True)
    conv = add_loop(conv, "if orow_i == 0:_ #1", "orow_o", 2, guard=True)
    conv = add_loop(conv, "if orow_o == 0:_ #1", "b", 4, guard=True)
    # Start fissioning loops
    conv = add_loop(conv, "for och_o in _:_ #0", "b", 4)
    conv = old_reorder(conv, "orow_o b")
    conv = old_reorder(conv, "orow_i b")
    conv = old_reorder(conv, "kcol b")
    conv = old_reorder(conv, "krow b")
    conv = fuse(conv, "for b in _:_ #0", "for b in _:_ #1")
    conv = fuse(conv, "for b in _:_ #0", "for b in _:_ #1", unsafe_disable_check=True)
    conv = fuse(conv, "for b in _:_ #0", "for b in _:_ #1")
    conv = fuse(conv, "for b in _:_ #0", "for b in _:_ #1", unsafe_disable_check=True)
    conv = add_loop(conv, "for och_o in _:_ #0", "ocol_o", 3)
    conv = old_reorder(conv, "orow_o ocol_o")
    conv = old_reorder(conv, "orow_i ocol_o")
    conv = old_reorder(conv, "kcol ocol_o")
    conv = old_reorder(conv, "krow ocol_o")
    conv = fuse(conv, "for ocol_o in _:_ #0", "for ocol_o in _:_ #1")
    conv = fuse(
        conv, "for ocol_o in _:_ #0", "for ocol_o in _:_ #1", unsafe_disable_check=True
    )
    conv = add_loop(conv, "for och_o in _:_ #0", "orow_o", 2)
    conv = old_reorder(conv, "orow_i orow_o")
    conv = old_reorder(conv, "kcol orow_o")
    conv = old_reorder(conv, "krow orow_o")
    conv = fuse(conv, "for orow_o in _:_ #0", "for orow_o in _:_ #1")
    conv = fuse(
        conv, "for orow_o in _:_ #0", "for orow_o in _:_ #1", unsafe_disable_check=True
    )
    conv = add_loop(conv, "for och_o in _:_ #0", "orow_i", 28)
    conv = old_reorder(conv, "kcol orow_i")
    conv = old_reorder(conv, "krow orow_i")
    conv = fuse(conv, "for orow_i in _:_ #0", "for orow_i in _:_ #1")
    conv = fuse(
        conv, "for orow_i in _:_ #0", "for orow_i in _:_ #1", unsafe_disable_check=True
    )

    conv = add_loop(conv, "for och_o in _:_ #3", "orow_o", 2)
    conv = fuse(conv, "for orow_o in _:_ #1", "for orow_o in _:_ #2")
    conv = fuse(
        conv, "for orow_o in _:_ #1", "for orow_o in _:_ #2", unsafe_disable_check=True
    )
    conv = add_loop(conv, "for och_o in _:_ #3", "orow_i", 28)
    conv = fuse(conv, "for orow_i in _:_ #1", "for orow_i in _:_ #2")
    conv = fuse(
        conv, "for orow_i in _:_ #1", "for orow_i in _:_ #2", unsafe_disable_check=True
    )

    conv = fuse(conv, "for krow in _:_ #0", "for krow in _:_ #1")
    conv = fuse(conv, "for kcol in _:_ #0", "for kcol in _:_ #1")
    conv = fuse(conv, "for krow in _:_ #1", "for krow in _:_ #2")
    conv = fuse(conv, "for kcol in _:_ #1", "for kcol in _:_ #2")

    conv = add_unsafe_guard(conv, "ld_i8_block_id2(_) #0", "orow_i == 0 or krow == 2")
    conv = add_unsafe_guard(conv, "ld_i8_block_id2(_) #1", "orow_i == 0 or krow == 2")

    conv = old_split(conv, "orow_i", 7, ["orow_io", "orow_ii"], perfect=True)
    conv = old_unroll(conv, "och_o")
    # conv = old_unroll(conv, 'kch_o')
    # conv = old_unroll(conv, 'kcol')
    conv = simplify(conv)
    print("after optimization:")
    print(conv)
    return conv, cpu


# schedule_conv_3()
my_schedule_conv_example()
