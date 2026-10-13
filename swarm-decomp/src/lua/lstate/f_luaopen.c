#include "lua.h"
#include "lstate.h"
#include "fn_luaH_new.h"
#include "fn_luaH_resize.h"
#include "fn_luaS_init.h"
#include "fn_luaT_init.h"
#include "fn_luaX_init.h"
#include "stack_init.h"

static void f_luaopen (lua_State *L, void *ud) {
  global_State *g = L->l_G;
  Table *registry;
  TValue *array;
  (void)ud;
  stack_init(L, L);
  registry = luaH_new(L);
  g->l_registry.value_.gc = (GCObject *)registry;
  g->l_registry.tt_ = 0x45;
  luaH_resize(L, registry, 2, 0);
  array = registry->array;
  array[0].value_.gc = (GCObject *)L;
  array[0].tt_ = 0x48;
  {
    GCObject *o = (GCObject *)luaH_new(L);
    array[1].tt_ = 0x45;
    array[1].value_.gc = o;
  }
  luaS_init(L);
  luaT_init(L);
  luaX_init(L);
  g->gcstp = 0;
  g->nilvalue.tt_ = 0;
}
