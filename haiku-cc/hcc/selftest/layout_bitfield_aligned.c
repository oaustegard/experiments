// EXPECT-EXIT: 0
// A bit-field whose declaration opens a new storage unit takes the member's
// __attribute__((aligned(N))) as its alignment (ms_struct layout, GCC PR target/52991).
extern void abort(void);
struct S {
  int a : 2;
  int b : 2;
  int c : 28;
  __attribute__((aligned(16))) int d : 2;
} __attribute__((ms_struct));
struct S s;
int main(void) {
  int i;
  if (sizeof(s) != 32)
    abort();
  s.d = -1;
  for (i = 0; i < 32; ++i)
    if (((signed char *)&s)[i] != (i == 16 ? 3 : 0))
      abort();
  return 0;
}
