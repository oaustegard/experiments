#include "systime.h"
#include "lauxlib.h"

#define SIZETIMEFMT 250

static int os_date (lua_State *L) {
  size_t slen;
  const char *s = luaL_optlstring(L, 1, "%c", &slen);
  time_t t = luaL_opt(L, l_checktime, 2, time(NULL));
  const char *se = s + slen;
  struct tm tmr, *stm;
  if (*s == '!') {
    stm = gmtime_r(&t, &tmr);
    s++;
  }
  else
    stm = localtime_r(&t, &tmr);
  if (stm == NULL)
    return luaL_error(L, "date result cannot be represented in this installation");
  if (strcmp(s, "*t") == 0) {
    lua_createtable(L, 0, 9);
    setallfields(L, stm);
  }
  else {
    char cc[4];
    luaL_Buffer b;
    cc[0] = '%';
    luaL_buffinit(L, &b);
    while (s < se) {
      if (*s != '%')
        luaL_addchar(&b, *s++);
      else {
        size_t reslen;
        char *buff = luaL_prepbuffsize(&b, SIZETIMEFMT);
        s++;
        s = checkoption(L, s, se - s, cc + 1);
        reslen = strftime(buff, SIZETIMEFMT, cc, stm);
        luaL_addsize(&b, reslen);
      }
    }
    luaL_pushresult(&b);
  }
  return 1;
}
