// EXPECT-EXIT: 0
#ifdef __NO_MATH_ERRNO__
#error "defined too early"
#endif
#pragma GCC optimize "-fno-math-errno"
#ifndef __NO_MATH_ERRNO__
#error "not defined"
#endif
#pragma GCC optimize "-fmath-errno"
#ifdef __NO_MATH_ERRNO__
#error "still defined"
#endif
int main(void) { return 0; }
