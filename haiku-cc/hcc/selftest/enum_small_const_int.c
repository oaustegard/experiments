// EXPECT-EXIT: 0
// Constants of an enum whose values all fit in int stay int, even though the
// enum's underlying type is unsigned: `i >= FOO` must go false at -1 (gcc-torture enum-2.c).
enum foo { FOO, BAR };
_Static_assert(((typeof(FOO))-1) < 0, "FOO is int");
int main(void) { int i, n = 0; for (i = BAR; i >= FOO; --i) n++; return n == 2 ? 0 : 1; }
