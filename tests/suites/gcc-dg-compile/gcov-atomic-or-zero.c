/* Test case to check that no atomic bitwise-or operations with a zero operand
   are emitted by instrument_decisions().  */
/* { dg-do compile } */
/* { dg-options "-O2 -fprofile-update=atomic -fcondition-coverage -fdump-tree-optimized" } */
/* { dg-require-effective-target profile_update_atomic } */

int a (void);
int b (void);
int c (int);

int
f (int *p)
{
  /* Force a condition chain so tree-profile's decision instrumentation kicks in.  */
  if (c (p[0]) || c (p[1]) || c (p[2]) || c (p[3]))
    return a ();
  else
    return b ();
}

/* We expect __atomic_fetch_or_() calls for condition counters.  */
/* { dg-final { scan-tree-dump "__atomic_fetch_or_" "optimized" } } */

/* After the optimization, no atomic bitwise-or operations with a zero
   operand should be present.  Match ", 0, 0)" in the __atomic_fetch_or_() call
   argument list.  */
/* { dg-final { scan-tree-dump-not "__atomic_fetch_or_. \\(\[^\n\r]*, 0, 0\\)" "optimized" } } */
