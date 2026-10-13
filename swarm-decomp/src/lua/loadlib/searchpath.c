#include <string.h>

#include "lauxlib.h"
#include "searchpath.h"
#include "getnextfilename.h"
#include "readable.h"

static const char *searchpath (lua_State *L, const char *name,
                               const char *path, const char *sep,
                               const char *dirsep) {
  luaL_Buffer buff;
  char *pathname;  /* path with name inserted */
  char *endpathname;  /* its end */
  luaL_Buffer errbuff;
  const char *filename;
  /* separator is non-empty and appears in 'name'? */
  if (*sep != '\0' && strchr(name, *sep) != NULL)
    name = luaL_gsub(L, name, sep, dirsep);  /* replace it by 'dirsep' */
  luaL_buffinit(L, &buff);
  /* add path to the buffer, replacing marks ('?') with the file name */
  luaL_addgsub(&buff, path, LUA_PATH_MARK, name);
  luaL_addchar(&buff, '\0');
  pathname = luaL_buffaddr(&buff);  /* writable list of file names */
  endpathname = pathname + luaL_bufflen(&buff) - 1;
  while ((filename = getnextfilename(&pathname, endpathname)) != NULL) {
    if (readable(filename))  /* does file exist and is readable? */
      return lua_pushstring(L, filename);  /* save and return name */
  }
  luaL_pushresult(&buff);  /* push path to create error message */
  {
    const char *msg = lua_tolstring(L, -1, NULL);
    luaL_buffinit(L, &errbuff);
    luaL_addstring(&errbuff, "no file '");
    luaL_addgsub(&errbuff, msg, LUA_PATH_SEP, "'\n\tno file '");
    luaL_addstring(&errbuff, "'");
    luaL_pushresult(&errbuff);
  }
  return NULL;  /* not found */
}
