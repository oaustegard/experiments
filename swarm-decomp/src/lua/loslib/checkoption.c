#include <string.h>
#include "lauxlib.h"

#define LUA_STRFTIMEOPTIONS \
  "aAbBcCdDeFgGhHIjmMnprRStTuUVwWxXyYzZ%" \
  "||" "EcECExEXEyEY" \
  "OdOeOHOIOmOMOSOuOUOVOwOWOy"

static const char *checkoption (lua_State *L, const char *conv, ptrdiff_t convlen, char *buff) {
  const char *option = LUA_STRFTIMEOPTIONS;
  int oplen = 1;
  for (; *option != '\0' && oplen <= convlen; option += oplen) {
    if (*option == '|')
      oplen++;
    else if (memcmp(conv, option, oplen) == 0) {
      memcpy(buff, conv, oplen);
      buff[oplen] = '\0';
      return conv + oplen;
    }
  }
  luaL_argerror(L, 1,
    lua_pushfstring(L, "invalid conversion specifier '%%%s'", conv));
  return conv;
}
