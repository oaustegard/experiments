#include "lua.h"
#include "lstate.h"
#include "LX.h"
#include "fn_luaF_closeupval.h"
#include "fn_luaM_free_.h"
#include "freestack.h"

void luaE_freethread(lua_State *L, lua_State *L1) {
  LX *l = (LX *)((char *)L1 - 8);
  luaF_closeupval(L1, L1->stack);
  freestack(L1);
  luaM_free_(L, l, sizeof(LX));
}
