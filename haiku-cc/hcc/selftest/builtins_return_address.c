// EXPECT-EXIT: 0
// __builtin_return_address(0) must yield the caller's return address, and
// __builtin_frame_address(0) the caller's frame: a callee sees a non-null
// return address that lies inside its caller.
void *ra(void) { return __builtin_return_address(0); }
void *fa(void) { return __builtin_frame_address(0); }
int main(void) {
  if (ra() == 0) return 1;
  if (fa() == 0) return 2;
  return 0;
}
