#include <stdlib.h>
#include <string.h>
#include "lua.h"
#include "lauxlib.h"
#include "setpath.h"

static void setpath (lua_State *L, const char *fieldname, const char *envname,
                                   const char *dft) {
  const char *dftmark;
  const char *nver = lua_pushfstring(L, "%s%s", envname, "_5_4");
  const char *path = getenv(nver);
  if (path == NULL)  /* no versioned environment variable? */
    path = getenv(envname);  /* try unversioned name */
  if (path == NULL || noenv(L))
    lua_pushstring(L, dft);  /* use default */
  else if ((dftmark = strstr(path, ";;")) == NULL)
    lua_pushstring(L, path);
  else {
    size_t len = strlen(path);
    luaL_Buffer b;
    luaL_buffinit(L, &b);
    if (path < dftmark) {  /* is there a prefix before ';;' ? */
      luaL_addlstring(&b, path, dftmark - path);  /* add it */
      luaL_addchar(&b, ';');
    }
    luaL_addstring(&b, dft);  /* add default culture */
    if (dftmark < path + len - 2) {  /* is there a suffix after ';;' ? */
      luaL_addchar(&b, ';');
      luaL_addlstring(&b, dftmark + 2, (path + len - 2) - dftmark);
    }
    luaL_pushresult(&b);
  }
  lua_setfield(L, -3, fieldname);  /* package[fieldname] = path value */
  lua_settop(L, -2);  /* pop the path value */
}
