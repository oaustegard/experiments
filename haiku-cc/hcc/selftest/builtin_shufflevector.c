// EXPECT-EXIT: 0
typedef int v4 __attribute__((vector_size(16)));
typedef double V1 __attribute__((vector_size(8)));
typedef double V2 __attribute__((vector_size(16)));
int main(void) {
  v4 a = {1, 2, 3, 4}, b = {5, 6, 7, 8};
  v4 r = __builtin_shufflevector(a, b, 7, 0, 4, 2);
  if (r[0] != 8 || r[1] != 1 || r[2] != 5 || r[3] != 3) return 1;
  V1 x = {3.5};
  V2 y = __builtin_shufflevector(x, (V2){}, 0, 0);
  if (y[0] != 3.5 || y[1] != 3.5) return 2;
  return 0;
}
