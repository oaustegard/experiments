// EXPECT-EXIT: 7
// Compatible redeclarations (incomplete array then complete, unprototyped then
// prototyped function) must still be accepted.
extern int a[];
int a[3];
int g();
int g(int x) { return x; }
int main(void) { extern int a[]; return g(7); }
