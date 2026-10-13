#include "lua.h"
#include "lstate.h"

void luaE_warning(lua_State *L, const char *msg, int tocont) {
  lua_WarnFunction wf = L->l_G->warnf;
  if (wf != NULL)
    wf(L->l_G->ud_warn, msg, tocont);
}
