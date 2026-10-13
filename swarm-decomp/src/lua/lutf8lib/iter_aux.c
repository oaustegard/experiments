#include "lauxlib.h"
#include "lutf8lib_common.h"

static int iter_aux (lua_State *L, int strict) {
  size_t len;
  const char *s = luaL_checklstring(L, 1, &len);
  size_t n = lua_tointeger(L, 2);
  if (n < len) {
    while (iscontp(s + n)) n++;  /* go to next character */
  }
  if (n >= len)
    return 0;  /* no more codepoints */
  else {
    utfint code;
    const char *next = utf8_decode(s + n, &code, strict);
    if (next == NULL || iscontp(next))
      return luaL_error(L, MSGInvalid);
    lua_pushinteger(L, n + 1);
    lua_pushinteger(L, code);
    return 2;
  }
}
