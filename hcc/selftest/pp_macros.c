// EXPECT-EXIT: 0
#include <limits.h>
#include <stddef.h>
#define SQ(x) ((x) * (x))
#define CAT(a, b) a##b
#define STR(x) #x
#define VA(fmt, ...) f(fmt, ##__VA_ARGS__)
#if defined(__x86_64__) && __STDC_VERSION__ >= 201112L && INT_MAX == 2147483647
#define OK 1
#else
#define OK 0
#endif
int f(int a, ...) { return a; }
int main(void) {
    int CAT(x, y) = SQ(3);
    if (xy != 9) return 1;
    if (sizeof(STR(a + b)) != 6) return 2;
    if (!OK) return 3;
    if (VA(4) != 4 || VA(5, 6, 7) != 5) return 4;
    if (__LINE__ != 20) return 5;
    if (sizeof(size_t) != 8) return 6;
    return 0;
}
