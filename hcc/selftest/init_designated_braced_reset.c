// EXPECT-EXIT: 0
// A designator followed by a braced list re-initializes the whole sub-object:
// the earlier brace-initialized string/int are dropped and the rest is zeroed.
extern void abort(void);
struct H { char I[6]; int J; };
struct H m[] = { { { "foo" }, 1 }, [0] = { .I[0] = 'b' } };
int main(void) {
  if (__builtin_memcmp(m[0].I, "b\0\0\0\0", 6) || m[0].J)
    abort();
  return 0;
}
