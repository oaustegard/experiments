// EXPECT-EXIT: 0
// A redeclaration that names the first member type of a transparent_union parameter is
// compatible with it (GCC accepts both orders).
typedef union { int *i; long *l; } U __attribute__((transparent_union));
extern void g(U);
extern void g(int *);
int main(void) { return 0; }
