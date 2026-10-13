// EXPECT-EXIT: 0
/* __builtin_<fn>f128 / acosh / fdim yield their libm return types, so a later
   redeclaration with the matching type is not a conflict. */
extern _Float128 t128;
extern __typeof (__builtin_fabsf128 (0)) t128;
extern __typeof (__builtin_acosf128 (0)) t128;
extern double td;
extern __typeof (__builtin_acosh (0)) td;
extern __typeof (__builtin_fdim (0, 0)) td;
extern __typeof (__builtin_ldexp (0, 0)) td;
int main(void) { return 0; }
