#include "lauxlib.h"
#include "lutf8lib_common.h"
#include "iter_aux.h"

static int iter_auxlax (lua_State *L) {
  return iter_aux(L, 0);
}
