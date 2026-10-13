#include <stdio.h>
#include "lua.h"
#include "lauxlib.h"
#include "warnf.h"

static void warnfon(void *ud, const char *message, int tocont) {
  if (checkcontrol((lua_State *)ud, message, tocont))
    return;
  fprintf(stderr, "%s", "Lua warning: ");
  fflush(stderr);
  warnfcont(ud, message, tocont);
}
