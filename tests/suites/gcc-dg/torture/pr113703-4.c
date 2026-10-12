/* { dg-do run } */
/* { dg-additional-options "-fwrapv-pointer -fno-tree-vectorize -fno-tree-loop-distribute-patterns" } */

#include <stdint.h>

uintptr_t sum = 0;

__attribute__((noipa)) void
f (char *p, unsigned int i, unsigned int n)
{
  p += i;
  do
    {
      sum += (uintptr_t)p;
      p += 1;
      i++;
    }
  while (i < n);
}

int
main ()
{
  /* Mirror image of gcc.dg/torture/ivopts-lt-3.c with an increasing pointer:
     the loop iterates 6 times and P walks past the end of the address space
     instead of its start.  */
  f ((char *)-4, 2, 8);
  /* SUM is -2 - 1 + 0 + 1 + 2 + 3.  */
  if (sum != 3)
    __builtin_abort ();
  return 0;
}
