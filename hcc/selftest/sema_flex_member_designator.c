// EXPECT-EXIT: 0
// A designator into a trailing flexible array member sets that element.
struct T { int a; char b[]; };
struct T t = { .a = 1, .b[2] = 'a' };
int main(void) {
  if (t.a != 1) return 1;
  if (t.b[2] != 'a' || t.b[0] != 0) return 2;
  return 0;
}
