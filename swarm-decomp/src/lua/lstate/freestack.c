#include "lua.h"
#include "lstate.h"
#include "fn_luaM_free_.h"
#include "freeCI.h"
#include "freestack.h"

static void freestack(lua_State *L) {
  if (L->stack == NULL)
    return;
  L->ci = &L->base_ci;
  freeCI(L);
  int n = (int)(L->stack_last - L->stack) + 5;
  luaM_free_(L, L->stack, n * sizeof(TValue));
}
