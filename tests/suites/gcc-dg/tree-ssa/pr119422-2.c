/* { dg-do run } */
/* { dg-options "-O1" } */

/* Macro adapted from builtin-object-size-common.h  */
#define FAIL() \
  do { \
    __builtin_printf ("Failure at line: %d\n", __LINE__);     \
    abort();						      \
  } while (0)

void abort(void);

int gg (_Bool b, unsigned a)
{
  return ((unsigned)b >= a) >= a;
}
int gg2 (_Bool b, unsigned a)
{
  return ((unsigned)b >= a) < a;
}

int main (void) {
  if (gg (0, 0) != 1)
    FAIL ();
  if (gg (1, 0) != 1)
    FAIL ();
  if (gg (1, 1) != 1)
    FAIL ();
  if (gg (0, 1) != 0)
    FAIL ();

  if (gg2 (0, 0) != 0)
    FAIL ();
  if (gg2 (1, 0) != 0)
    FAIL ();
  if (gg2 (1, 1) != 0)
    FAIL ();
  if (gg2 (0, 1) != 1)
    FAIL ();

  return 0;
}
