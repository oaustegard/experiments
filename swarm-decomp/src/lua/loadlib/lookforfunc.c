#include <dlfcn.h>
#include "lua.h"
#include "lauxlib.h"

#define CLIBS "_CLIBS"
#define ERRLIB 1
#define ERRFUNC 2
#define l_unlikely(x) __builtin_expect(((x) != 0), 0)

static int lookforfunc (lua_State *L, const char *path, const char *sym) {
  void *reg;
  lua_getfield(L, LUA_REGISTRYINDEX, CLIBS);
  lua_getfield(L, -1, path);
  reg = lua_touserdata(L, -1);
  lua_settop(L, -3);
  if (l_unlikely(reg == NULL)) {
    reg = dlopen(path, RTLD_NOW | (*sym == '*' ? RTLD_GLOBAL : 0));
    if (l_unlikely(reg == NULL)) {
      lua_pushstring(L, dlerror());
      return ERRLIB;
    }
    lua_getfield(L, LUA_REGISTRYINDEX, CLIBS);
    lua_pushlightuserdata(L, reg);
    lua_pushvalue(L, -1);
    lua_setfield(L, -3, path);
    lua_rawseti(L, -2, luaL_len(L, -2) + 1);
    lua_settop(L, -2);
  }
  if (*sym == '*') {
    lua_pushboolean(L, 1);
    return 0;
  }
  else {
    lua_CFunction f = (lua_CFunction)dlsym(reg, sym);
    if (l_unlikely(f == NULL)) {
      lua_pushstring(L, dlerror());
      return ERRFUNC;
    }
    lua_pushcclosure(L, f, 0);
    return 0;
  }
}
