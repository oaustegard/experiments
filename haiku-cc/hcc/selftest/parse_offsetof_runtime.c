// EXPECT-EXIT: 0
// __builtin_offsetof with a non-constant array subscript is evaluated at run time.
#include <stddef.h>
struct A { int a; int b[10]; };
int get(struct A *p) { return (int) __builtin_offsetof(struct A, b[p->a]); }
int main(void) {
    struct A x;
    x.a = 3;
    return get(&x) == 3 * (int) sizeof(int) + (int) offsetof(struct A, b) ? 0 : 1;
}
