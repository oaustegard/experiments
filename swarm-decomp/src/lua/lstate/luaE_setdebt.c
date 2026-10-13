#include "lua.h"
#include "lstate.h"

#define MAX_LMEM ((l_mem)(~(lu_mem)0 >> 1))

void luaE_setdebt(global_State *g, l_mem debt) {
  l_mem tb = g->totalbytes + g->GCdebt;
  if (debt < tb - MAX_LMEM)
    debt = tb - MAX_LMEM;
  g->totalbytes = tb - debt;
  g->GCdebt = debt;
}
