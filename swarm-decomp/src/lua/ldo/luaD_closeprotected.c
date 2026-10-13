#include "CloseP.h"
#include "Pfunc.h"
#include "fn_luaD_closeprotected.h"

static void closepaux (lua_State *L, void *ud);
int luaD_rawrunprotected (lua_State *L, Pfunc f, void *ud);

int luaD_closeprotected (lua_State *L, ptrdiff_t level, int status) {
  CallInfo *old_ci = L->ci;
  lu_byte old_allowhooks = L->allowhook;
  for (;;) {
    struct CloseP pcl;
    pcl.level = (StkId)((char *)L->stack + level);
    pcl.status = status;
    status = luaD_rawrunprotected(L, closepaux, &pcl);
    if (__builtin_expect(status == 0, 1))
      return pcl.status;
    else {
      L->ci = old_ci;
      L->allowhook = old_allowhooks;
    }
  }
}
