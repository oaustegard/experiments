// EXPECT-EXIT: 0
typedef unsigned int u32 __attribute__((aligned(4), warn_if_not_aligned(8)));
struct s { int x; } __attribute__((warn_if_not_aligned(8)));
int main(void) { u32 a = 1; return a == 1 ? 0 : 1; }
