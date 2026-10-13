#include <string.h>
#include "lua.h"
#include "lstate.h"
#include "fn_luaC_newobjdt.h"
#include "fn_luaC_step.h"
#include "stack_init.h"

lua_State *lua_newthread (lua_State *L) {
  global_State *g = L->l_G;
  lua_State *L1;
  if (g->GCdebt > 0)
    luaC_step(L);
  L1 = (lua_State *)luaC_newobjdt(L, 8, 0xd0, 8);
  L->top->value_.gc = (GCObject *)L1;
  L->top->tt_ = 0x48;
  L->top++;
  L1->l_G = g;
  L1->stack = NULL;
  L1->ci = NULL;
  L1->nci = 0;
  L1->twups = L1;
  L1->nCcalls = 0;
  L1->errorJmp = NULL;
  L1->hook = NULL;
  L1->hookmask = 0;
  L1->basehookcount = 0;
  L1->allowhook = 1;
  L1->hookcount = L1->basehookcount;
  L1->openupval = NULL;
  L1->status = 0;
  L1->errfunc = 0;
  L1->oldpc = 0;
  L1->hookmask = L->hookmask;
  L1->basehookcount = L->basehookcount;
  L1->hook = L->hook;
  L1->hookcount = L1->basehookcount;
  memcpy(lua_getextraspace(L1), lua_getextraspace(g->mainthread), LUA_EXTRASPACE);
  stack_init(L1, L);
  return L1;
}
