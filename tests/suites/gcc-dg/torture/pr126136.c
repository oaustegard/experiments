/* { dg-do run } */

int i;

__attribute__ ((noipa,noclone,noinline))
signed char foo(signed char val)
{
  i++;
  if (i > 1)
    return -1;
  else
    return val;
}

int
main(void)
{
  i = 0;
  if (foo (10) != 10)
    __builtin_abort ();
  i += 2;
  if (foo (10) != -1)
    __builtin_abort ();
}
