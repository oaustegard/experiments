#include "lauxlib.h"
#include "g_iofile.h"

static int io_output (lua_State *L) {
  return g_iofile(L, "_IO_output", "w");
}
