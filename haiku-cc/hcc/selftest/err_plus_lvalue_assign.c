// EXPECT-ERROR: lvalue required as left operand of assignment
// Unary '+' yields an rvalue: assigning to +x is invalid.
int main(void)
{
  int x = 0;
  +x = 1;
  return x;
}
