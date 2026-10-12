// EXPECT-EXIT: 0
// #pragma GCC optimize toggles the -f math flag macros (rounding, trapping,
// reciprocal, associative, signed-zeros) on and off.
#pragma GCC optimize "-frounding-math"
#ifndef __ROUNDING_MATH__
#error "__ROUNDING_MATH__ not defined"
#endif
#pragma GCC optimize "-fno-rounding-math"
#ifdef __ROUNDING_MATH__
#error "__ROUNDING_MATH__ defined"
#endif
#pragma GCC optimize "-fno-trapping-math"
#ifndef __NO_TRAPPING_MATH__
#error "__NO_TRAPPING_MATH__ not defined"
#endif
#pragma GCC optimize "-freciprocal-math"
#ifndef __RECIPROCAL_MATH__
#error "__RECIPROCAL_MATH__ not defined"
#endif
#pragma GCC optimize "-fassociative-math"
#ifndef __ASSOCIATIVE_MATH__
#error "__ASSOCIATIVE_MATH__ not defined"
#endif
#pragma GCC optimize "-fno-signed-zeros"
#ifndef __NO_SIGNED_ZEROS__
#error "__NO_SIGNED_ZEROS__ not defined"
#endif
#pragma GCC optimize "-fsigned-zeros"
#ifdef __NO_SIGNED_ZEROS__
#error "__NO_SIGNED_ZEROS__ defined"
#endif
int main(void) { return 0; }
