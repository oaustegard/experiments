#include "systime.h"
#include "lauxlib.h"

static int os_time (lua_State *L) {
  time_t t;
  if (lua_isnoneornil(L, 1))
    t = time(NULL);
  else {
    struct tm ts;
    luaL_checktype(L, 1, LUA_TTABLE);
    lua_settop(L, 1);
    ts.tm_year = getfield(L, "year", -1, 1900);
    ts.tm_mon = getfield(L, "month", -1, 1);
    ts.tm_mday = getfield(L, "day", -1, 0);
    ts.tm_hour = getfield(L, "hour", 12, 0);
    ts.tm_min = getfield(L, "min", 0, 0);
    ts.tm_sec = getfield(L, "sec", 0, 0);
    ts.tm_isdst = getboolfield(L, "isdst");
    t = mktime(&ts);
    setallfields(L, &ts);
  }
  if (t == (time_t)(-1))
    return luaL_error(L, "time result cannot be represented in this installation");
  lua_pushinteger(L, (lua_Integer)t);
  return 1;
}
