// EXPECT-STDOUT: 6 2.5
#include <stdarg.h>
#include <stdio.h>
int sum(int n, ...) { va_list ap; va_start(ap, n); int s = 0; for (int i = 0; i < n; i++) s += va_arg(ap, int); va_end(ap); return s; }
double avg(int n, ...) { va_list ap; va_start(ap, n); double s = 0; for (int i = 0; i < n; i++) s += va_arg(ap, double); va_end(ap); return s / n; }
int main(void) { printf("%d %.1f\n", sum(3, 1, 2, 3), avg(2, 2.0, 3.0)); return 0; }
