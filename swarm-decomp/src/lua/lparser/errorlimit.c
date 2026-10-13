#include "FuncState.h"
#include "fn_luaO_pushfstring.h"
#include "fn_luaX_syntaxerror.h"

static void errorlimit (FuncState *fs, int limit, const char *what) {
  lua_State *L = fs->ls->L;
  const char *msg;
  int line = fs->f->linedefined;
  const char *where = (line == 0)
                      ? "main function"
                      : luaO_pushfstring(L, "function at line %d", line);
  msg = luaO_pushfstring(L, "too many %s (limit is %d) in %s", what, limit, where);
  luaX_syntaxerror(fs->ls, msg);
}
