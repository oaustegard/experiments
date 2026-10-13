#include "lua.h"
#include "lstate.h"

static void close_state(lua_State *L);

void lua_close(lua_State *L) {
  L = L->l_G->mainthread;
  close_state(L);
}
