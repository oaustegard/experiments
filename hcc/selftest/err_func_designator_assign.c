// EXPECT-ERROR: lvalue required as left operand of assignment
// A function designator is not a modifiable lvalue.
int f(void);
int main(void)
{
  f = 0;
  return 0;
}
