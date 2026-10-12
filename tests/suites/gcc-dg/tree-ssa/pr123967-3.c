/* { dg-do run } */
/* { dg-additional-options -O2 } */

void abort(void);

/* Macro adapted from builtin-object-size-common.h  */
#define FAIL() \
  do { \
    __builtin_printf ("Failure at line: %d\n", __LINE__);     \
    abort();						      \
  } while (0)

unsigned f1 (unsigned x, int m)
{
  int mask = 0x3;

  if (x)
    m |= mask;
  else
    m = m & ~mask;

  return m + 1;
}

unsigned f2 (unsigned x, int m)
{
  if (x)
    m |= 0x3;
  else
    m = m & ~0x8;

  return m + 1;
}

int main (void)
{
  if (f1 (0, 7) != 5)
    FAIL ();
  if (f1 (1, 7) != 8)
    FAIL ();

  if (f2 (0, 8) != 1)
    FAIL ();
  if (f2 (1, 8) != 12)
    FAIL ();    

  return 0;
}
