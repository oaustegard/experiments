// EXPECT-EXIT: 9
// Compound assignment and ++/-- on an _Atomic int are read-modify-write updates.
int main(void) {
  _Atomic int x = 3;
  int a = x++;          /* a=3, x=4 */
  x += 4;               /* x=8 */
  int b = ++x;          /* b=9, x=9 */
  x -= 1;               /* x=8 */
  x |= 1;               /* x=9 */
  return a + b - 3 + (x - 9) + 0 * x == 9 ? 9 : 0;
}
