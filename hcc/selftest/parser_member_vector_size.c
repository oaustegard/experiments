// EXPECT-EXIT: 0
// vector_size written after a struct member's declarator makes it a vector.
struct S { double s __attribute__((vector_size(16))); int k; };
int main(void) {
  struct S x = { { 1.0, 2.0 }, 3 };
  if (sizeof(x) != 32) return 1;
  return x.s[1] == 2.0 && x.k == 3 ? 0 : 2;
}
