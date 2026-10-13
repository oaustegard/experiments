#include "lua.h"
#include "lauxlib.h"
#include "warnf.h"

static void warnfoff(void *ud, const char *message, int tocont) {
  checkcontrol((lua_State *)ud, message, tocont);
}
