#include "propagatemark.h"

static lu_mem propagateall(global_State *g) {
  lu_mem work = 0;
  while (g->gray)
    work += propagatemark(g);
  return work;
}
