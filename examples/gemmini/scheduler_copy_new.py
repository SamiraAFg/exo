import os
import sys
from abc import ABC, abstractmethod
from exo.core.LoopIR import LoopIR
from exo.core.LoopIR import T as TT
import exo.API_cursors as pc
from exo.libs.memories import GEMM_SCRATCH, GEMM_ACCUM
from exo import proc, instr, DRAM, config, ExoType
from exo.stdlib.scheduling import *
from exo.stdlib.stdlib import *
from exo.stdlib.inspection import *

from gemmini_schedules import *

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class MatMulMapping:
    # order: tuple[str]
    # load : tuple[int]
    # i: tuple[int]
    # j: tuple[int]
    # k: tuple[int]
    # workload_size: tuple[int, int, int]

    def __init__(self, order, loads, i, j, k, workload_size):
        self.order = order
        self.i = i
        self.j = j
        self.k = k
        self.workload_size = workload_size
        self.loads = loads
        # validate order
        assert isinstance(self.order, tuple), "order must be a tuple"
        assert all(isinstance(x, str) for x in self.order), "order must contain strings"

        # validate I, J, K and loads
        assert len(loads) == 3, "determine the loop level for each load [A, B, O]"
        for name, v in [
            ("loads", self.loads),
            ("i", self.i),
            ("j", self.j),
            ("k", self.k),
        ]:
            assert isinstance(v, tuple), f"{name} must be a tuple"
            assert all(isinstance(x, int) for x in v), f"{name} must contain integers"
        dim = [1, 1, 1]
        for val in self.i:
            dim[0] *= val
        for val in self.j:
            dim[1] *= val
        for val in self.k:
            dim[2] *= val
        assert (
            dim[idx] == self.workload_size[idx] for idx in range(3)
        ), "product of loop bounds doesn't match the workload size (N(i), M(j), K(k))"

        # A Dimension
        i_loops = i
        k_loops = k

        for loop in order[0 : loads[0] + 1]:
            if loop.startswith("i"):
                i_loops = i_loops[1:]
            elif loop.startswith("k"):
                k_loops = k_loops[1:]
        self.A_dim = i_loops + k_loops
        # B Dimension
        k_loops = k
        j_loops = j
        for loop in order[0 : loads[1] + 1]:
            if loop.startswith("j"):
                j_loops = j_loops[1:]
            elif loop.startswith("k"):
                k_loops = k_loops[1:]
        self.B_dim = k_loops + j_loops
        # print(f"A_dim: {self.A_dim}")
        print(f"B_dim: {self.B_dim}")
        # O Dimension
        i_loops = i
        j_loops = j
        for loop in order[0 : loads[2] + 1]:
            if loop.startswith("i"):
                i_loops = i_loops[1:]
            elif loop.startswith("j"):
                j_loops = j_loops[1:]

        self.O_dim = i_loops + j_loops

    def calculate_size(self, dim: tuple[int]):
        # Calculate buffer sizes
        size = 1
        for x in dim:
            size *= x
        return size

    @property
    def A_size(self):

        return self.calculate_size(self.A_dim)

    @property
    def B_size(self):
        return self.calculate_size(self.B_dim)

    @property
    def O_size(self):
        return self.calculate_size(self.O_dim)

    @property
    def outer_most_loop(self):
        loop = self.order[0]
        assert loop[0] in [
            "i",
            "j",
        ], "outer most loop must be  i or j, k is not supported yet!"
        return loop

    def get_A_dim_permute_vector(self, same=True, loops=None):
        if loops is None:
            loops = []
            for d in self.order[(self.loads[0] + 1) :]:
                if d.startswith("i") or d.startswith("k"):
                    loops.append(d)
        vec = [loops.index(l) for l in loops if l.startswith("i")] + [
            loops.index(l) for l in loops if l.startswith("k")
        ]
        if not same:
            vec[::2] = [loops.index(l) for l in loops if l.startswith("i")]
            vec[1::2] = [loops.index(l) for l in loops if l.startswith("k")]
        return vec

    def get_B_dim_permute_vector(self, same=True, loops=None):
        if loops is None:
            loops = []
            for d in self.order[(self.loads[1] + 1) :]:
                if d.startswith("k") or d.startswith("j"):
                    loops.append(d)
        vec = [loops.index(l) for l in loops if l.startswith("k")] + [
            loops.index(l) for l in loops if l.startswith("j")
        ]
        if not same:
            vec[::2] = [loops.index(l) for l in loops if l.startswith("k")]
            vec[1::2] = [loops.index(l) for l in loops if l.startswith("j")]
        return vec

    def get_ML_tiled_loops(self):
        loops = dict()
        if len(self.i) > 2:
            loops["i"] = self.i[::-1]

        if len(self.j) > 2:
            loops["j"] = self.j[::-1]

        if len(self.k) > 2:
            loops["k"] = self.k[::-1]
        return loops

    def get_order_wo_reduce(self):
        return [item for item in self.order if not item.startswith("k")]

    def validate(self, gemm_order: tuple, gemm_tvals):
        tvals = []
        for loop in gemm_order:
            if loop == "i":
                tvals.append(self.i[-1])
            elif loop == "j":
                tvals.append(self.j[-1])
            elif loop == "k":
                tvals.append(self.k[-1])
        order = [x[0] for x in self.order[-3:]]
        assert (order == list(gemm_order)) and (
            tvals == list(gemm_tvals)
        ), "This mapping is not valid because it doesn't match the hardware instructions"

    def find_guard_loops(self):
        # extract top-level loops
        num = max(self.loads) + 1
        # k_flag = False
        top_loops = self.order[:num]
        # if top_loops[0].startswith('k'):
        #     k_flag = True
        top_loops2 = []
        k_loops = []
        for i in range(num):
            if not top_loops[i].startswith("k"):
                top_loops2.append(top_loops[i])
            else:
                k_loops.append(top_loops[i])
        if k_loops is not None:
            top_loops2.extend(k_loops)
        print(top_loops2)
        # A guards
        aguards = []
        anum = self.loads[0] + 1
        if anum == 0:
            if top_loops2[0].startswith("j"):
                aguards.append(top_loops2[0])
        else:
            for l in reversed(top_loops2[:anum]):
                if l.startswith("k") or l.startswith("i"):
                    break
                else:
                    aguards.append(l)
        # B guards
        bguards = []
        bnum = self.loads[1] + 1
        if bnum == 0:
            if top_loops2[0].startswith("i"):
                bguards.append(top_loops2[0])
        else:
            for l in reversed(top_loops2[:bnum]):
                if l.startswith("k") or l.startswith("j"):
                    break
                else:
                    bguards.append(l)
        return aguards, bguards


class Scheduler(ABC):
    def __init__(self, mapping: MatMulMapping):
        # self.proc = proc
        self.mapping = mapping
        super().__init__()

    @abstractmethod
    def get_load_a_pat(self):
        pass

    @abstractmethod
    def get_load_b_pat(self):
        pass

    @abstractmethod
    def get_init_out_pat(self):
        pass

    @abstractmethod
    def get_gemm_pat(self):
        pass

    @abstractmethod
    def get_store_pat(self):
        pass

    def set_cursors(self, proc):
        self.i_loop = proc.find_loop("i")
        self.j_loop = proc.find_loop("j")
        self.k_loop = proc.find_loop("k")
        self.res_load = proc.find("res = D[_]")
        self.a_assign = proc.find("a2 = A[_]")
        self.b_assign = proc.find("b2 = B[_]")
        self.res_alloc = self.res_load.prev()
        self.out_assign = proc.find("C[_, _] = _")

    def get_val(self, loop: str, order: tuple[str], vals: tuple[int]):

        return vals[order.index(loop)]

    def get_cur_pat(self, cursor, num=2):
        size = num
        loops = []
        parent = cursor.parent()
        while size > 0:
            loops.append(parent)
            parent = parent.parent()
            size -= 1
        res = loops[::-1]
        assert len(res) == num, f" there are only {len(res)} loops not {num}"
        return res

    def interleaved_rearrange_vec(self, cursor, buffer: str):
        assert buffer == "A" or buffer == "B", "buffer must be 'A' or 'B'"
        second_idx = "k" if buffer == "A" else "j"
        idxs = idx_list(cursor)
        print(idxs)
        idx_size = len(idxs)
        assert (
            idx_size > 2
        ), f"interleaved_rearrange doesn't mean for a buffer of size {idx_size}"
        for idx in idxs:
            if idx.startswith(second_idx):
                fidx_size = idxs.index(idx)
                break
        sidx_size = idx_size - fidx_size
        vec = [i for i in range(idx_size)]
        out = vec.copy()
        if fidx_size == sidx_size:
            out[::2] = vec[:fidx_size]
            out[1::2] = vec[fidx_size:]
        elif fidx_size > sidx_size:
            if sidx_size == 1:
                return out
            assert fidx_size - sidx_size <= 1, "NOT SUPPORTED! TO BE ADDED (if needed)"
            out[1::2] = vec[1:fidx_size]
            out[2::2] = vec[fidx_size:]
        else:
            if fidx_size == 1:
                out[: idx_size - 2] = vec[1 : idx_size - 1]
                out[-2] = vec[0]
            assert sidx_size - fidx_size <= 1, "NOT SUPPORTED! TO BE ADDED (if needed)"
            out[::2] = vec[fidx_size:]
            out[1::2] = vec[:fidx_size]
        return out

    def multiply_buffer_loops2(self, proc, loop_list):
        # print(proc)
        dnum = len(loop_list)
        assert dnum > 2, "only 2 loops! multiplication only for the same loops"
        first_idx = loop_list[0].name()[0]
        loop_list_o, loop_list_i = [
            l for l in loop_list if l.name().startswith(first_idx)
        ], [l for l in loop_list if not l.name().startswith(first_idx)]
        # temp1 = [l.name() for l in loop_list_o]
        # temp2 = [l.name() for l in loop_list_i]
        # print(f"loop_list_o: {temp1}, loop_list_i: {temp2}")
        for i in range(len(loop_list_o) - 1):
            lo = loop_list_o[-(i + 2)]
            try:
                proc = simplify(mult_loops(proc, lo, lo.name()[0]))
            except:
                break

        for i in range(len(loop_list_i) - 1):
            li = loop_list_i[-(i + 2)]
            try:
                proc = simplify(mult_loops(proc, li, li.name()[0]))
            except:
                break

        # print(f"after mult2: {simplify(proc)}")
        return simplify(proc)

    def multiply_buffer_loops(self, proc, loop_list):
        # print(proc)
        dnum = len(loop_list)
        assert dnum > 2, "only 2 loops! multiplication only for the same loops"
        loop_list_o, loop_list_i = [l for l in loop_list[: dnum // 2]], [
            l for l in loop_list[dnum // 2 :]
        ]

        if dnum % 2 == 0:
            for i in range((dnum // 2) - 1):
                li = loop_list_i[-(i + 2)]
                lo = loop_list_o[-(i + 2)]
                proc = mult_loops(proc, li, li.name()[0])
                proc = mult_loops(proc, lo, lo.name()[0])
        else:
            repeat_i = dnum // 2
            repeat_o = dnum // 2 - 1
            for i in range(repeat_i):
                li = loop_list_i[-(i + 2)]
                proc = mult_loops(proc, li, li.name()[0])
            if repeat_o != 0:
                for i in range(repeat_o - 1):
                    lo = loop_list_o[-(i + 2)]
                    proc = mult_loops(proc, lo, lo.name()[0])

        # print(f"after mult1: {simplify(proc)}")
        return simplify(proc)

    def set_pattern_and_replace_mem(
        self,
        proc,
        cursor,
        pat_order,
        pat_vals,
        alloc_cursor,
        dst_mem,
        instr_tuple,
        blocked=False,
    ):
        # print("-"*40)
        # print(proc)
        loop_cursors = self.get_cur_pat(proc.forward(cursor))
        c_order = [l.name() for l in loop_cursors]
        c_vals = [l.hi().value() for l in loop_cursors]
        cc_order = [x[0] for x in c_order]
        if cc_order != list(pat_order):
            proc = reorder_loops(proc, " ".join(c_order))
            # n_order = c_order[::-1]
            n_vals = c_vals[::-1]
            n_loops = loop_cursors[::-1]
        else:
            # n_order = c_order
            n_vals = c_vals
            n_loops = loop_cursors

        def low_level_tiling(proc, loops, vals, pvals, blocked=False):
            # enable_tiling = [x >= y if blocked else x > y for x, y in zip(vals, pvals)] #there is a bug here for blocked transfers without 1 in pattern values
            enable_tiling = [x > y for x, y in zip(vals, pvals)]
            lnames = [l.name() for l in loops]
            ldict = {lnames[0]: 1, lnames[1]: 1}
            if enable_tiling[0]:
                proc, _ = tile_loops(proc, [(loops[0], pvals[0])])
                ldict[lnames[0]] = 2
            if enable_tiling[1]:
                proc, _ = tile_loops(proc, [(loops[1], pvals[1])])
                proc = lift_scope(proc, loops[1])
                ldict[lnames[1]] = 2
            # print(f"ldict: {ldict}")
            proc = simplify(proc)
            print(f"proc after low_level_tiling: {proc}")
            return proc, ldict

        if not blocked:
            # enable_tiling = [x <= y for x, y in zip(n_vals, pat_vals)]

            # if not enable_tiling[0]:
            #     proc, _ = tile_loops(proc, [(n_loops[0], pat_vals[0])])

            # if not enable_tiling[1]:
            #     proc, _ = tile_loops(proc, [(n_loops[1], pat_vals[1])])
            #     proc = lift_scope(proc, n_loops[1])
            proc, _ = low_level_tiling(proc, n_loops, n_vals, pat_vals)
        else:
            assert (
                len(pat_vals) == 4
            ), "For blocked memory transfer the pattern values have to be 4 [io, jo, ii, ji] OR [jo, io, ji, ii]"
            l0 = n_loops[0]
            l1 = n_loops[1]
            # proc, _ = tile_loops(proc, [(l0, pat_vals[-2]), (l1, pat_vals[-1])])
            proc, ldic = low_level_tiling(
                proc, n_loops, n_vals[-2:], pat_vals[2:], blocked=True
            )
            c_loops = self.get_cur_pat(proc.forward(cursor), sum(ldic.values()))
            # print(f"ldic: {ldic}")
            tmp_vals = [1, 1]
            if sum(ldic.values()) < 4:
                for i, n in enumerate(ldic.values()):
                    if n != 1:
                        tmp_vals[i] = c_loops[0].hi().value()
                c_vals = tmp_vals + [l.hi().value() for l in c_loops[-2:]]
            else:
                c_vals = [l.hi().value() for l in c_loops]
            print(f"c_vals after blocked tiling: {c_vals}")
            if 1 in pat_vals[:2]:
                if pat_vals.index(1) == 0 and c_vals[1] > pat_vals[1]:
                    proc, _ = tile_loops(proc, [(l1, pat_vals[1])])
                    # elif c_vals[1] == 1:
                    #     proc = add_loop(proc, proc.find_loop("ki"), f"{l1.name()[0]}o", 1)
                elif pat_vals.index(1) == 1 and c_vals[0] > pat_vals[0]:
                    proc = lift_scope(proc, l1)
                    proc, _ = tile_loops(proc, [(l0, pat_vals[0])])
            elif not all([x <= y for x, y in zip(c_vals[:2], pat_vals[:2])]):
                proc, _ = tile_loops(proc, [(l0, pat_vals[0]), (l1, pat_vals[1])])

        proc = simplify(proc)
        print(proc)
        if dst_mem != DRAM:
            proc = set_memory(proc, alloc_cursor, dst_mem)
        proc = simplify(replace_and_inline(proc, instr_tuple))
        return proc

    def multi_loop_tiling(self, proc):
        # proc, _ = self.first_level_tiling(proc)
        # out_intr_order, _ = self.get_init_out_pat
        self.set_cursors(proc)
        intr_order, intr_vals = self.get_gemm_pat()
        self.mapping.validate(intr_order, intr_vals)
        # reorder = self.mapping.outer_most_loop == 'j' TODO: if you get kij working remove the assertion error in outer_most_loop
        i_low_val = self.get_val("i", intr_order, intr_vals)
        j_low_val = self.get_val("j", intr_order, intr_vals)
        k_low_val = self.get_val("k", intr_order, intr_vals)

        proc, [ii, ji] = tile_loops(
            proc, [(self.i_loop, i_low_val), (self.j_loop, j_low_val)]
        )

        proc, [ki] = tile_loops(proc, [(self.k_loop, k_low_val)])
        loops_dict = self.mapping.get_ML_tiled_loops()

        if loops_dict:
            for loop, num in loops_dict.items():
                name = f"{loop}o"
                c = proc.find_loop(name)
                for n in range(
                    1, len(num) - 1
                ):  # starting from 1,  because inner loops are already tiled
                    proc = divide_loop(
                        proc, c, num[n], [f"{name}o", f"{name}i"], perfect=True
                    )  # TODO make sure the c moves to the outer loop automatically
                    name = f"{name}o"
        return proc

    def replace_init_out(self, proc, mem_size: int, instr_tuple: tuple, dst_mem):

        order = self.mapping.get_order_wo_reduce()
        proc = reorder_loops_from_idx_zz(proc, self.res_alloc, order)

        proc, _ = autolift_alloc(proc, self.res_alloc, max_size=mem_size)
        proc = fission_as_much_as_possible(proc, self.res_load)
        intr_order, intr_vals = self.get_init_out_pat()

        proc = self.set_pattern_and_replace_mem(
            proc,
            self.res_load,
            intr_order,
            intr_vals,
            self.res_alloc,
            dst_mem,
            instr_tuple,
        )
        return proc

    def apply_mapping_order(self, proc):
        proc = fission_as_much_as_possible(proc, self.k_loop)
        proc = reorder_loops_from_idx_zz(proc, self.a_assign, list(self.mapping.order))
        return proc

    def replace_load_buffers(
        self,
        proc,
        a_dst_mem,
        b_dst_mem,
        a_instr_tuple,
        b_instr_tuple,
        mem_size: int,
        blocked=True,
    ):
        # Step 1: lift allocation
        (proc, a_size), a_load, a_alloc = bind_and_lift(
            proc, proc.forward(self.a_assign).rhs(), max_size=self.mapping.A_size
        )
        (proc, b_size), b_load, b_alloc = bind_and_lift(
            proc, proc.forward(self.b_assign).rhs(), max_size=self.mapping.B_size
        )
        print(proc)
        # print(f"A_size: {a_size}, B_size: {b_size}, mem_size: {mem_size}")
        assert a_size + b_size <= mem_size, "The allocated size is exceeded!"
        # print("after lift alloc:")
        # print(proc)
        proc = reorder_top(proc, a_alloc)
        proc = reorder_top(proc, b_alloc)

        # Step 2: fission
        proc = fission_as_much_as_possible(proc, a_load)
        proc = fission_as_much_as_possible(proc, b_load)
        print("after fission:")
        print(proc)

        # Step 2.5: remove redundant loops
        proc = remove_redundant_loops(
            proc, a_load, num=2
        )  # TODO: Bug: for more than three nested loops 2 is not enough
        proc = remove_redundant_loops(proc, b_load, num=2)

        # Step 3: restructure for pattern matching
        # a) interleaved reordering
        A_perm_vec = self.mapping.get_A_dim_permute_vector(
            loops=idx_list(proc.forward(a_load))
        )
        B_perm_vec = self.mapping.get_B_dim_permute_vector(
            loops=idx_list(proc.forward(b_load))
        )
        # print(f"idx_list A: {idx_list(proc.forward(a_load))}, idx_list B: {idx_list(proc.forward(b_load))}")
        # print(f"A_perm_vec: {A_perm_vec}, B_perm_vec: {B_perm_vec}")
        proc = rearrange_dim(proc, a_alloc, A_perm_vec)
        proc = rearrange_dim(proc, b_alloc, B_perm_vec)
        proc = reorder_loops_from_idx(proc, a_load)
        proc = reorder_loops_from_idx(proc, b_load)
        print("after rearrange and reorder:")
        print(proc)
        # b) unifying the loops
        # print(idx_list_org(proc.forward(a_load)))
        A_loops = get_loops_at_or_above(proc.forward(a_load))[-len(A_perm_vec) :]
        # A_loops = get_loops_at_or_above(proc.forward(a_load))[-len(idx_list_org(proc.forward(a_load))):]
        print(f"loop_list for A load: {[l.name() for l in A_loops]}")
        proc = self.multiply_buffer_loops2(proc, A_loops)
        # proc = self.multiply_buffer_loops(proc, loop_list[-len(A_perm_vec):]) #only works if outer loops are less than inner loops one time!!

        B_loops = get_loops_at_or_above(proc.forward(b_load))[-len(B_perm_vec) :]
        # B_loops = get_loops_at_or_above(proc.forward(b_load))[-len(idx_list_org(proc.forward(b_load))):]
        # print(f"loop_list for B load: {[l.name() for l in B_loops]}")
        proc = self.multiply_buffer_loops2(proc, B_loops)
        # proc = self.multiply_buffer_loops(proc, loop_list[-len(B_perm_vec):])
        print("after mult:")
        print(proc)

        ## test for dim more than 4
        # a_idx = idx_list(proc.forward(a_load))
        # b_idx = idx_list(proc.forward(b_load))
        # first_aidx = a_idx[0][0]
        # first_bidx = b_idx[0][0]
        # aidx_list_o, aidx_list_i = [d for d in a_idx if d[0] == first_aidx], [d for d in a_idx if d[0] != first_aidx]
        # bidx_list_o, bidx_list_i = [d for d in b_idx if d[0] == first_bidx], [d for d in b_idx if d[0] != first_bidx]
        # print(f"aidx_list_o: {aidx_list_o}, aidx_list_i: {aidx_list_i}")
        # print(f"bidx_list_o: {bidx_list_o}, bidx_list_i: {bidx_list_i}")
        # for i in range(len(aidx_list_o) - 2):
        #     proc = simplify(mult_dim(proc, a_alloc, 0, 1))
        # for i in range(len(aidx_list_i) - 2):
        #     proc = simplify(mult_dim(proc, a_alloc, 2, 3))
        # for i in range(len(bidx_list_o) - 2):
        #     proc = simplify(mult_dim(proc, b_alloc, 0, 1))
        # for i in range(len(bidx_list_i) - 2):
        #     proc = simplify(mult_dim(proc, b_alloc, 2, 3))
        # print("after rearrange and reorder2:")
        # print(proc)
        # Step 4: tiling to match the instruction patterns
        A_perm_vec = self.interleaved_rearrange_vec(proc.forward(a_load), "A")
        B_perm_vec = self.interleaved_rearrange_vec(proc.forward(b_load), "B")

        proc = rearrange_dim(proc, a_alloc, A_perm_vec)
        proc = rearrange_dim(proc, b_alloc, B_perm_vec)
        # print("after rearrange_dim:")
        # print(proc)
        a_intr_order, a_intr_vals = self.get_load_a_pat()
        proc = self.set_pattern_and_replace_mem(
            proc,
            a_load,
            a_intr_order,
            a_intr_vals,
            a_alloc,
            a_dst_mem,
            a_instr_tuple,
            blocked,
        )
        b_intr_order, b_intr_vals = self.get_load_b_pat()
        proc = self.set_pattern_and_replace_mem(
            proc,
            b_load,
            b_intr_order,
            b_intr_vals,
            b_alloc,
            b_dst_mem,
            b_instr_tuple,
            blocked,
        )

        return proc

    def replace_gemm(self, proc, instr_tuple):
        # the gemm pattern has been validated at first.
        proc = simplify(replace_and_inline(proc, instr_tuple))
        return proc

    def replace_store(self, proc, instr_tuple):
        intr_order, intr_vals = self.get_store_pat()
        proc = self.set_pattern_and_replace_mem(
            proc,
            self.out_assign,
            intr_order,
            intr_vals,
            self.res_alloc,
            DRAM,
            instr_tuple,
        )
        return proc

    def add_guards(self, proc):
        aguards, bguards = self.mapping.find_guard_loops()
        print(aguards, bguards)
        if aguards is not None:
            for l in aguards:
                print(f"{l} == 0")
                ac = proc.find("for _ in _:_").find("_(_) #1")
                proc = add_unsafe_guard(proc, ac, f"{l} == 0")
        if bguards is not None:
            for l in bguards:
                bc = proc.find("for _ in _:_").find("_(_) #2")
                proc = add_unsafe_guard(proc, bc, f"{l} == 0")
        return proc

    # def add_guards_new(self, proc):
    #     top_loops = []
    #     def recursive_add_loop(cursor):
    #         if isinstance(cursor, pc.ForCursor):
    #             if len(cursor.body()) == 1 and isinstance(cursor.body()[0], pc.ForCursor):
    #                 top_loops.append(cursor)
    #                 recursive_add_loop(cursor.body()[0])
    #     recursive_add_loop(proc)
    def add_guards_new(self, proc):
        top_loops = []

        def recursive_add_loop(cursor):
            if isinstance(cursor, pc.ForCursor):
                top_loops.append(cursor)
                if len(cursor.body()) > 1:
                    return
                recursive_add_loop(cursor.body()[0])

        recursive_add_loop(proc.find("for _ in _:_"))


# <<<<<<<<<<<<< Schedule Abstract Functions >>>>>>>>>>>>>>>>>>
def get_loops_at_or_above(cursor):
    loops = []
    while isinstance((parent := cursor.parent()), pc.ForCursor):
        loops.append(parent)
        cursor = parent
    return list(reversed(loops))
