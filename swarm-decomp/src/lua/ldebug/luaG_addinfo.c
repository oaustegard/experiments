#include "lstate.h"
#include "fn_luaO_chunkid.h"
#include "fn_luaO_pushfstring.h"
#include "fn_luaG_addinfo.h"

#define getstr(ts)	((ts)->contents)
#define tsslen(s)	((s)->shrlen != 0xff ? (s)->shrlen : (s)->u.lnglen)

const char *luaG_addinfo (lua_State *L, const char *msg, TString *src, int line) {
  char buff[LUA_IDSIZE];
  if (src)
    luaO_chunkid(buff, getstr(src), tsslen(src));
  else {  /* no source available; use "?" instead */
    buff[0] = '?'; buff[1] = '\0';
  }
  return luaO_pushfstring(L, "%s:%d: %s", buff, line, msg);
}
