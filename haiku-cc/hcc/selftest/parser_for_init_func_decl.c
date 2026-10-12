// EXPECT-ERROR: non-variable
void foo(void)
{
  for (int i = 1, bar (void); i <= 10; i++)
    ;
}
