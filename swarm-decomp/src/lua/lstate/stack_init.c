#include "lua.h"
#include "lstate.h"
#include "fn_luaM_malloc_.h"

static void stack_init (lua_State *L1, lua_State *L) {
  int i;
  CallInfo *ci;
  L1->stack = (StkId)luaM_malloc_(L, (2 * LUA_MINSTACK + 5) * sizeof(TValue), 0);
  L1->tbclist = L1->stack;
  for (i = 0; i < 2 * LUA_MINSTACK + 5; i++)
    L1->stack[i].tt_ = 0;
  L1->top = L1->stack;
  L1->stack_last = L1->stack + 2 * LUA_MINSTACK;
  ci = &L1->base_ci;
  ci->next = ci->previous = NULL;
  ci->callstatus = 2;
  ci->func = L1->top;
  ci->u.c.k = NULL;
  ci->nresults = 0;
  L1->top->tt_ = 0;
  L1->top++;
  ci->top = L1->top + LUA_MINSTACK;
  L1->ci = ci;
}
