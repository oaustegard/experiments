#include "lstate.h"
#include "fn_luaD_reallocstack.h"
#include "fn_luaD_throw.h"
#include "fn_luaG_runerror.h"
#include "fn_luaD_growstack.h"

#define ERRORSTACKSIZE (LUAI_MAXSTACK + 200)

int luaD_growstack (lua_State *L, int n, int raiseerror) {
  int size = (int)(L->stack_last - L->stack);
  if (__builtin_expect((size > LUAI_MAXSTACK), 0)) {
    if (raiseerror)
      luaD_throw(L, LUA_ERRERR);
    return 0;
  }
  else if (n < LUAI_MAXSTACK) {
    int newsize = 2 * size;
    int needed = (int)(L->top - L->stack) + n;
    if (newsize > LUAI_MAXSTACK)
      newsize = LUAI_MAXSTACK;
    if (newsize < needed)
      newsize = needed;
    if (__builtin_expect((newsize <= LUAI_MAXSTACK), 1))
      return luaD_reallocstack(L, newsize, raiseerror);
  }
  luaD_reallocstack(L, ERRORSTACKSIZE, raiseerror);
  if (raiseerror)
    luaG_runerror(L, "stack overflow");
  return 0;
}
