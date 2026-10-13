#include "traverseCclosure.h"

static int traverseCclosure(global_State *g, CClosure *cl) {
  int i;
  for (i = 0; i < cl->nupvalues; i++)
    markvalue(g, &cl->upvalue[i]);
  return 1 + cl->nupvalues;
}
