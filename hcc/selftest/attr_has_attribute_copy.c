// EXPECT-EXIT: 7
// __builtin_has_attribute sees noreturn, also when inherited via copy(f);
// a bare __builtin_<libc> names the library function as a value.
#define ASRT(expr) _Static_assert(expr, #expr)
__attribute__((noreturn)) void fn(void);
__attribute__((copy(fn))) void fn_copy(void);
ASRT(__builtin_has_attribute(fn, noreturn));
ASRT(__builtin_has_attribute(fn_copy, noreturn));
ASRT(!__builtin_has_attribute(fn, leaf));
int main(void) {
  int (*p)(const char *, ...) = 1 ? __builtin_printf : 0;
  return p == 0 ? 1 : 7;
}
