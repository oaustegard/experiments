#include "lauxlib.h"
#include "lutf8lib_common.h"
#include "iter_aux.h"

static int iter_auxstrict (lua_State *L) {
  return iter_aux(L, 1);
}
