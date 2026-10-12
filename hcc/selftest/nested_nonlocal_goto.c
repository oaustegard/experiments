// EXPECT-STDOUT: 6
#include <stdio.h>
int execute(int cmd)
{
  __label__ start;
  int depth = 5;
  void raise(void) { goto start; }
  int last;
  last = 0;
start:
  if (last == 0) while (1) { last = 1; raise(); }
  return cmd + depth;
}
int main(void) { printf("%d\n", execute(1)); return 0; }
