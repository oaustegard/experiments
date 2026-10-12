// EXPECT-ERROR: expected expression before
void f(int x)
{
  if (x > 1)
    int ret = 1;
}
