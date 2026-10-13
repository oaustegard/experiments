#include "lstate.h"
#include "fn_luaF_close.h"
#include "lua.h"

static StkId index2stack (lua_State *L, int idx);

LUA_API void lua_closeslot (lua_State *L, int idx) {
  StkId level;
  level = index2stack(L, idx);
  level = luaF_close(L, level, -1, 0);
  level->tt_ = 0;
}
