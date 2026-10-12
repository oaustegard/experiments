// EXPECT-ERROR: non-variable
void foo(void)
{
  for (typedef int T;;)
    ;
}
