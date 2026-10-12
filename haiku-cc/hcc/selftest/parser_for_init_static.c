// EXPECT-ERROR: static variable
void foo(void)
{
  for (static int i = 1; i <= 10; i++)
    ;
}
