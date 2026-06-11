#include "test_gemmini_matmul_new_bias_zz.h"

#include <stdio.h>
#include <stdlib.h>

#include <include/gemmini.h>
#include "gemm_acc_malloc.h"
float _select_float(float x,float v,float y,float z) {
    if (x < v) return y;
    else return z;
}

// clamp(
//     src : f32 @DRAM,
//     dst : i8 @DRAM
// )
void clamp( test_gemmini_matmul_new_bias_zz_Context *ctxt, const float* src, int8_t* dst ) {
float l;
float h;
l = -128.0f;
h = 127.0f;
float tmp;
tmp = _select_float((float)h, (float)*src, (float)h, (float)*src);
tmp = _select_float((float)*src, (float)l, (float)l, (float)tmp);
*dst = (int8_t)(tmp);
}


/* relying on the following instruction..."
config_st_acc_i8(scale,dst_stride,act)
gemmini_extended_config_st({dst_stride}, {act}, {scale}[0]);

*/

/* relying on the following instruction..."
do_st_acc_i8(n,m,src,dst)
gemmini_extended_mvout( ((uint64_t) &{dst_data}), (uint32_t) &{src_data}, {m}, {n} );
*/
// st_acc_i8_v2(
//     n : size,
//     m : size,
//     scale : f32 @DRAM,
//     act : bool,
//     src : [i32][n, 16] @GEMM_ACCUM,
//     dst : [i8][n, m] @DRAM
// )
void st_acc_i8_v2( test_gemmini_matmul_new_bias_zz_Context *ctxt, int_fast32_t n, int_fast32_t m, const float* scale, bool act, struct exo_win_2i32c src, struct exo_win_2i8 dst ) {
EXO_ASSUME(n <= 16);
EXO_ASSUME(m <= 16);
// assert stride(dst, 1) == 1
// assert stride(src, 0) == 16
// assert stride(src, 1) == 1
gemmini_extended_config_st((dst.strides[0]), (act), (scale)[0]);

gemmini_extended_mvout( ((uint64_t) &dst.data[0]), (uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)src)) + (0)/16)), (m + 0), (n + 0) );
}

