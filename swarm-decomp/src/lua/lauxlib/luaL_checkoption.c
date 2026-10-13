#include <string.h>
#include "lauxlib.h"

int luaL_checkoption (lua_State *L, int arg, const char *def, const char *const lst[]) {
  const char *name;
  if (def == NULL)
    name = luaL_checkstring(L, arg);
  else if (lua_type(L, arg) > 0)
    name = luaL_checkstring(L, arg);
  else
    name = def;
  int i;
  for (i=0; lst[i]; i++)
    if (strcmp(lst[i], name) == 0)
      return i;
  return luaL_argerror(L, arg,
                       lua_pushfstring(L, "invalid option '%s'", name));
}
