// EXPECT-EXIT: 0
// <float.h> reads the gcc-predefined __FLT32_* macros (FLT32_ names from float.h).
#define __STDC_WANT_IEC_60559_TYPES_EXT__
#include <float.h>
#ifndef FLT32_MANT_DIG
#error "FLT32_MANT_DIG undefined"
#endif
#if FLT32_MANT_DIG != 24 || FLT32_DIG != 6
#error "bad FLT32 values"
#endif
int main(void) { return 0; }
