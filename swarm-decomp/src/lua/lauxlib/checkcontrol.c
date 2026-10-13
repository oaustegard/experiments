#include <string.h>
#include "lua.h"
#include "lauxlib.h"
#include "warnf.h"

static int checkcontrol(lua_State *L, const char *message, int tocont) {
  if (tocont || *(message++) != '@')
    return 0;
  else {
    if (strcmp(message, "off") == 0)
      lua_setwarnf(L, warnfoff, L);
    else if (strcmp(message, "on") == 0)
      lua_setwarnf(L, warnfon, L);
    return 1;
  }
}
