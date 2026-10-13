// EXPECT-EXIT: 0
// Before C23, a C11 Annex D.1 character may begin an identifier (gcc -std=c99..gnu17
// accepts these; gcc-dg/ucnid-15-utf8.c). C23 is strict XID (reject-gcc-dg/cpp/ucnid-16-utf8.c).
int ² = 2;
int 😀x = 3;
int main(void) { return ² + 😀x == 5 ? 0 : 1; }
