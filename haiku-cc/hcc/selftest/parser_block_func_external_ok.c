// EXPECT-EXIT: 5
// A later file-scope declaration with a matching type makes z visible: no error.
void f(void) { long z(); }
long z(void) { return 5; }
void g(void) { z(); }
int main(void) { return (int)z(); }
