#include "lauxlib.h"
#include "g_iofile.h"

static int io_input (lua_State *L) {
  return g_iofile(L, "_IO_input", "r");
}
