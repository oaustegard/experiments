// EXPECT-EXIT: 0
// #pragma pack caps natural member alignment; push/pop restore the outer value.
struct A { char c; double d; };
#pragma pack(push, p1, 1)
struct B { char c; double d; };
#pragma pack(pop, p1)
struct C { char c; double d; };
#pragma pack(2)
struct D { char c; int i; };
#pragma pack()
int a1[(sizeof(struct A) == 16) ? 1 : -1];
int a2[(sizeof(struct B) == 9) ? 1 : -1];
int a3[(sizeof(struct C) == 16) ? 1 : -1];
int a4[(sizeof(struct D) == 6) ? 1 : -1];
int main(void) { return 0; }
