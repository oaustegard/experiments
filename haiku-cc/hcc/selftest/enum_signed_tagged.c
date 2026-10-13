// EXPECT-EXIT: 0
// A tagged enum with a negative enumerator is signed int: it is compatible with int
// (so a redeclaration of a function returning int is accepted) and sizeof is 4.
enum E { E1 = -1, E2 = 0, E3 = 1 };
int foo(void);
enum E foo(void) { return E2; }
int main(void) {
    return sizeof(enum E) == 4 ? 0 : 1;
}
