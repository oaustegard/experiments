// EXPECT-EXIT: 0
// Bit-fields of _Bool, enum and integer types remain valid.
enum E { Q };
struct S { _Bool b : 1; enum E e : 3; unsigned long long z : 40; int : 0; int x : 4; };
int main(void) { struct S s = {0}; return 0; }
