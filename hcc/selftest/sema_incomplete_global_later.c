// EXPECT-EXIT: 0
// Tentative file-scope object whose struct type is completed later is fine.
struct s1 v1;
struct s1 { int x; };
static int a[];
int main(void) { v1.x = 0; return 0; }
