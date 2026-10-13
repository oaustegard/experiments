#include "setpause.h"
#include "fn_luaE_setdebt.h"

static void setpause(global_State *g) {
  l_mem threshold, debt;
  int pause = g->gcpause * 4;
  l_mem estimate = g->GCestimate / 100;
  threshold = (pause < 0x7fffffffffffffffLL / estimate) ? estimate * pause : 0x7fffffffffffffffLL;
  debt = (g->totalbytes + g->GCdebt) - threshold;
  if (debt > 0)
    debt = 0;
  luaE_setdebt(g, debt);
}
