// EXPECT-EXIT: 0
// A stray non-UTF-8 byte outside literals vanishes; `a` then sees `)` and expands.
#define a b(
#define b(x) q
int aœ);
int main(void) { return sizeof(q) == sizeof(int) ? 0 : 1; }
