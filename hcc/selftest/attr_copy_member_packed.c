// EXPECT-EXIT: 0
// copy(expr) on a member inherits `packed` from the expression's struct type;
// __builtin_has_attribute sees packed on such members.
struct P { char c; int i; } __attribute__((packed));
struct Q { __attribute__((copy((struct P *)0))) short m; char z; int k; };
#define ASRT(expr) _Static_assert(expr, #expr)
ASRT(__builtin_has_attribute(((struct Q*)0)->m, packed));
ASRT(!__builtin_has_attribute(((struct Q*)0)->k, packed));
int main(void) { return 0; }
