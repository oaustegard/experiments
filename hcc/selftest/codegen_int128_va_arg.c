// EXPECT-EXIT: 0
// EXPECT-STDOUT: 40 3
#include <stdarg.h>
#include <stdio.h>
typedef __int128 s128;
static s128 sum(int n, ...) { va_list ap; va_start(ap, n); s128 t = 0; for (int i = 0; i < n; i++) t += va_arg(ap, s128); va_end(ap); return t; }
int main(void) { s128 r = sum(3, (s128)1 << 70, (s128)5, (s128)-2); printf("%llx %llx\n", (unsigned long long)(r >> 64), (unsigned long long)r); return 0; }
