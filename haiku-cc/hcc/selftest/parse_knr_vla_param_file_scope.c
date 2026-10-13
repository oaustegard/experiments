// EXPECT-EXIT: 0
// A K&R parameter declaration may be variably modified (gcc-dg-compile/Wshadow-3.c):
// `int w[v];` between `)` and `{` is not a file-scope declaration. (Known gap: the size
// here binds the global `v`; hcc does not yet bring K&R parameters into scope for later
// K&R declarations, which GCC does.)
int v = 2;
int f8(n, w) int n; int w[v]; { return w[n - 1]; }
int main(void) { int a[2] = {0, 0}; return f8(2, a); }
