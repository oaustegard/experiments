#include "setminordebt.h"
#include "fn_luaE_setdebt.h"

static void setminordebt(global_State *g) {
  luaE_setdebt(g, -(((lu_mem)(g->totalbytes + g->GCdebt) / 100) * g->genminormul));
}
