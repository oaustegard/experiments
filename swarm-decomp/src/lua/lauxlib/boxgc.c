#include "lauxlib.h"
#include "resizebox.h"

static int boxgc (lua_State *L) {
  resizebox(L, 1, 0);
  return 0;
}
