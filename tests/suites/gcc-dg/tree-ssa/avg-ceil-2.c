/* Execution test for the rounded-average fold
     ((x >> 1) + (y >> 1)) + ((x | y) & 1)  ->  (x | y) - ((x ^ y) >> 1)
   at the extremes of the signed and unsigned ranges.  The subtraction in
   the folded form cannot overflow: its result is the ceiling average,
   which always lies between min (x, y) and max (x, y).  Compare against
   the source expression evaluated in a wider type.  */
/* { dg-do run } */
/* { dg-options "-O2" } */
/* { dg-require-effective-target int32plus } */

#define INT_MIN (-__INT_MAX__ - 1)
#define INT_MAX __INT_MAX__

__attribute__ ((noipa)) int
avg_ceil_signed (int x, int y)
{
  return ((x >> 1) + (y >> 1)) + ((x | y) & 1);
}

__attribute__ ((noipa)) unsigned
avg_ceil_unsigned (unsigned x, unsigned y)
{
  return ((x >> 1) + (y >> 1)) + ((x | y) & 1u);
}

int
main (void)
{
  static const int sv[] = { INT_MIN, INT_MIN + 1, -3, -2, -1, 0, 1, 2, 3,
			    INT_MAX - 1, INT_MAX };
  static const unsigned uv[] = { 0, 1, 2, 3, 0x7ffffffe, 0x7fffffff,
				 0x80000000u, 0xfffffffeu, 0xffffffffu };
  unsigned i, j;

  for (i = 0; i < sizeof sv / sizeof sv[0]; i++)
    for (j = 0; j < sizeof sv / sizeof sv[0]; j++)
      {
	long long x = sv[i], y = sv[j];
	long long ref = ((x >> 1) + (y >> 1)) + ((x | y) & 1);
	if (avg_ceil_signed (sv[i], sv[j]) != (int) ref)
	  __builtin_abort ();
      }

  for (i = 0; i < sizeof uv / sizeof uv[0]; i++)
    for (j = 0; j < sizeof uv / sizeof uv[0]; j++)
      {
	unsigned long long x = uv[i], y = uv[j];
	unsigned long long ref = ((x >> 1) + (y >> 1)) + ((x | y) & 1);
	if (avg_ceil_unsigned (uv[i], uv[j]) != (unsigned) ref)
	  __builtin_abort ();
      }
  return 0;
}
