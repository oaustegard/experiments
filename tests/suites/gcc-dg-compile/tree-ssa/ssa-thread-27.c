/* Cover adjust_paths_after_duplication, which had no test.  Threading
   the path out of "t == 1" leaves two registered candidates starting on
   the edge it started on: one shares a prefix with it and is rewired
   onto the copies, the other is entirely a prefix of it and is dropped.  */
/* { dg-do compile } */
/* { dg-options "-O2 -fdump-tree-threadfull1-details" } */

int g;
void bar (void);

void
f (int a, int b)
{
  int t = 0;
  if (a > 0)
    t = 1;
  if (b > 0)
    t += 2;
  if (t == 1)
    bar ();
  if (t == 3)
    bar ();
  if (t == 0)
    bar ();
  g = t;
}

/* Both candidates are considered.  */
/* { dg-final { scan-tree-dump-times "adjusting candidate" 2 "threadfull1" } } */
/* One of them keeps a tail and is rewired onto the copied blocks.  */
/* { dg-final { scan-tree-dump "adjusted candidate" "threadfull1" } } */
/* The other has nothing left once the shared prefix goes.  */
/* { dg-final { scan-tree-dump "Adjusted candidate is EMPTY" "threadfull1" } } */
