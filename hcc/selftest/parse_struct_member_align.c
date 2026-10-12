// EXPECT-EXIT: 0
// Member aligned(N) survives a packed struct; tagged `struct S;` in a struct
// declares nothing; ms_struct bit-fields start after plain data.
struct A { char a; };
struct B { struct A; char b; };
struct P { char c; short s __attribute__((aligned(8))); } __attribute__((packed));
struct M { int a; short b; int c : 15; char d; } __attribute__((ms_struct));
int main(void) {
    if (sizeof(struct B) != 1) return 1;
    if (sizeof(struct P) != 16) return 2;
    if (__builtin_offsetof(struct P, s) != 8) return 3;
    if (sizeof(struct M) != 16) return 4;
    if (__builtin_offsetof(struct M, d) != 12) return 5;
    return 0;
}
