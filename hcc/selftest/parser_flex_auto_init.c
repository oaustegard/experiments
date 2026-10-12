// EXPECT-ERROR: [parse]
// Automatic-storage initialization of a flexible array member is rejected (gcc: non-static).
typedef char T[];
struct str { int len; T s; };
void f(void) {
    struct str c = { 2, "c" };
    (void)c;
}
