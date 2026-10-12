/* { dg-do run } */
/* { dg-additional-options "-fwrapv-pointer -fno-tree-vectorize -fno-tree-loop-distribute-patterns" } */

#include <stdint.h>

uintptr_t sum = 0;

__attribute__((noipa)) void
f (char *p, unsigned int i, unsigned int n)
{
  p -= i;
  do
    {
      sum += (uintptr_t)p;
      p -= 1;
      i++;
    }
  while (i < n);
}

int
main ()
{
  /* I is 2 and N is 8, so the loop iterates 6 times and P walks past the
     start of the address space, which is well defined here.  Its values
     are therefore not ordered like those of I and the counter cannot be
     eliminated in favour of it, but IVOPTs used to do it nevertheless and
     the loop exited at the first test.  */
  f ((char *)4, 2, 8);
  /* SUM is 2 + 1 + 0 - 1 - 2 - 3.  */
  if (sum != (uintptr_t)-3)
    __builtin_abort ();
  return 0;
}
