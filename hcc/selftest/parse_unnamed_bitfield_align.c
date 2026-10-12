// EXPECT-EXIT: 0
struct A { char c; int : 3; };
struct S { int : 0; };
union U { int : 0; };
union V { char c; int : 5; };
int main(void) {
    if (_Alignof(struct A) != 1) return 1;
    if (sizeof(struct A) != 2) return 2;
    if (_Alignof(struct S) != _Alignof(union U)) return 3;
    if (_Alignof(union V) != 1) return 4;
    return 0;
}
