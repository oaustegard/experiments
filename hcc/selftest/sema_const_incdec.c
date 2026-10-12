// EXPECT-ERROR: read-only
void f(const int n)
{
  n++;
}
int main(void)
{
  f(1);
  return 0;
}
