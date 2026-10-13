#include "incstep.h"
#include "singlestep.h"
#include "setpause.h"
#include "fn_luaE_setdebt.h"

static void incstep(lua_State *L, global_State *g) {
  {
    int stepmul = (g->gcstepmul * 4 | 1);
    l_mem debt = (g->GCdebt / sizeof(TValue)) * stepmul;
    l_mem stepsize = (g->gcstepsize <= 62)
                       ? (((l_mem)1 << g->gcstepsize) / sizeof(TValue)) * stepmul
                       : 0x7fffffffffffffffLL;
    do {
      lu_mem work = singlestep(L);
      debt -= work;
    } while (debt > -stepsize && g->gcstate != GCSpause);
    if (g->gcstate == GCSpause)
      setpause(g);
    else {
      debt = (debt / stepmul) * sizeof(TValue);
      luaE_setdebt(g, debt);
    }
  }
}
