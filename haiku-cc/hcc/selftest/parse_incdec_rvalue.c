// EXPECT-ERROR: lvalue required as increment operand
// Postfix ++ on a non-lvalue (an integer constant) must be rejected.
int main(void)
{
  int x = 7++;
  return x;
}
