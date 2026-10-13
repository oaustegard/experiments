#include "lauxlib.h"

static int test_eof (lua_State *L, FILE *f) {
  int c = getc(f);
  ungetc(c, f);
  lua_pushliteral(L, "");
  return (c != EOF);
}
