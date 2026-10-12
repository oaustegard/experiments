// EXPECT-EXIT: 0
struct s { int a; int x __attribute__((warn_if_not_aligned(16))); };
int main(void) { struct s v; v.x = 0; return 0; }
