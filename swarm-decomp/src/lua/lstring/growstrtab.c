#include <limits.h>
#include "lstate.h"
#include "TString.h"
#include "fn_luaC_fullgc.h"
#include "fn_luaD_throw.h"
#include "fn_luaS_resize.h"
#include "growstrtab.h"

#define MAXSTRTB INT_MAX
#define l_unlikely(x) __builtin_expect(((x) != 0), 0)

static void growstrtab (lua_State *L, stringtable *tb) {
  if (l_unlikely(tb->nuse == INT_MAX)) {
    luaC_fullgc(L, 1);
    if (tb->nuse == INT_MAX)
      luaD_throw(L, LUA_ERRMEM);
  }
  if (tb->size <= MAXSTRTB / 2)
    luaS_resize(L, tb->size * 2);
}
