// EXPECT-EXIT: 2
int main(void) {
  double r = __builtin_copysign(1.0, -0.0);
  if (r > 0.0) return 1;
  if (__builtin_copysign(2.0, -1.0) != -2.0) return 3;
  return 2;
}
