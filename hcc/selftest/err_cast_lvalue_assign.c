// EXPECT-ERROR: lvalue required as left operand of assignment
// A cast is never an lvalue, even to the same type.
int x;
int main(void)
{
  (int) x = 1;
  return 0;
}
