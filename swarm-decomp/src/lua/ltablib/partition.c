#pragma once
#include "auxsort.h"

static IdxT partition (lua_State *L, IdxT lo, IdxT up);
static IdxT partition (lua_State *L, IdxT lo, IdxT up) {
  IdxT i = lo;
  IdxT j = up - 1;
  for (;;) {
    while (lua_geti(L, 1, ++i), sort_comp(L, -1, -2)) {
      if (luai_unlikely(i == up - 1))
        luaL_error(L, "invalid order function for sorting");
      lua_settop(L, -2);
    }
    while (lua_geti(L, 1, --j), sort_comp(L, -3, -1)) {
      if (luai_unlikely(j < i))
        luaL_error(L, "invalid order function for sorting");
      lua_settop(L, -2);
    }
    if (j < i) {
      lua_settop(L, -2);
      set2(L, up - 1, i);
      return i;
    }
    set2(L, i, j);
  }
}
