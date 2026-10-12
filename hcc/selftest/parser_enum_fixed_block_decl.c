// EXPECT-EXIT: 7
// C23: `enum e : T;` in a block declares a new tag in that block, even when an
// outer `enum e` without a fixed underlying type exists.
enum e1 { E1A };
int main(void) {
  enum e1 : short;
  enum e1 : int { E1B = 7 };
  return E1B;
}
