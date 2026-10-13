#include "l_noret.h"
#include "fn_luaG_runerror.h"
#include "fn_luaT_objtypename.h"
l_noret luaG_forerror (lua_State *L, const TValue *o, const char *what) {
  luaG_runerror(L, "bad 'for' %s (number expected, got %s)",
                   what, luaT_objtypename(L, o));
}
