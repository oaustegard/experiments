#include <stdio.h>
#include "lua.h"
#include "lauxlib.h"
#include "warnf.h"

static void warnfcont(void *ud, const char *message, int tocont) {
  lua_State *L = (lua_State *)ud;
  fprintf(stderr, "%s", message);
  fflush(stderr);
  if (tocont)
    lua_setwarnf(L, warnfcont, L);
  else {
    fprintf(stderr, "%s", "\n");
    fflush(stderr);
    lua_setwarnf(L, warnfon, L);
  }
}
