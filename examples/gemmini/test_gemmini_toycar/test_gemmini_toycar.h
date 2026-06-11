
#pragma once
#ifndef TEST_GEMMINI_TOYCAR_H
#define TEST_GEMMINI_TOYCAR_H

#ifdef __cplusplus
extern "C" {
#endif


#include <stdint.h>
#include <stdbool.h>

// Compiler feature macros adapted from Hedley (public domain)
// https://github.com/nemequ/hedley

#if defined(__has_builtin)
#  define EXO_HAS_BUILTIN(builtin) __has_builtin(builtin)
#else
#  define EXO_HAS_BUILTIN(builtin) (0)
#endif

#if EXO_HAS_BUILTIN(__builtin_assume)
#  define EXO_ASSUME(expr) __builtin_assume(expr)
#elif EXO_HAS_BUILTIN(__builtin_unreachable)
#  define EXO_ASSUME(expr) \
      ((void)((expr) ? 1 : (__builtin_unreachable(), 1)))
#else
#  define EXO_ASSUME(expr) ((void)(expr))
#endif

typedef struct test_gemmini_toycar_Context { 

    struct ConfigStore {
        float scale;
        int_fast32_t dst_stride;
        bool act;
    } ConfigStore;

} test_gemmini_toycar_Context;
#ifndef EXO_WIN_2I32C
#define EXO_WIN_2I32C
struct exo_win_2i32c{
    const int32_t * const data;
    const int_fast32_t strides[2];
};
#endif
#ifndef EXO_WIN_2I8
#define EXO_WIN_2I8
struct exo_win_2i8{
    int8_t * const data;
    const int_fast32_t strides[2];
};
#endif
// clamp(
//     src : f32 @DRAM,
//     dst : i8 @DRAM
// )
void clamp( test_gemmini_toycar_Context *ctxt, const float* src, int8_t* dst );

// st_acc_i8_v2(
//     n : size,
//     m : size,
//     scale : f32 @DRAM,
//     act : bool,
//     src : [i32][n, 16] @GEMM_ACCUM,
//     dst : [i8][n, m] @DRAM
// )
void st_acc_i8_v2( test_gemmini_toycar_Context *ctxt, int_fast32_t n, int_fast32_t m, const float* scale, bool act, struct exo_win_2i32c src, struct exo_win_2i8 dst );



#ifdef __cplusplus
}
#endif
#endif  // TEST_GEMMINI_TOYCAR_H
