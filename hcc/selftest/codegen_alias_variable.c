// EXPECT-EXIT: 0
// A variable with __attribute__((alias)) names the target's storage.
static const int a[3] = { 4, 5, 6 };
static const int A[3] __attribute__((alias("a")));
int main(void) {
  return A[1] == 5 && A[2] == 6 ? 0 : 1;
}
