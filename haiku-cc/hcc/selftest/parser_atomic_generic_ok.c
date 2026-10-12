// EXPECT-EXIT: 0
// Generic __atomic_* forms accept lvalue arrays and pointer-valued calls as operands.
int b, c, d[4];
int *get(void) { return &c; }
int main(void) {
  __atomic_load(&b, &c, 0);
  __atomic_load(d, &c, 0);
  __atomic_store(get(), &b, 0);
  __atomic_exchange(&b, get(), &d[1], 0);
  return 0;
}
