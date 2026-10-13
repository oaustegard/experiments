#include "lua.h"
#include "lstate.h"
#include "fn_luaM_free_.h"
#include "freeCI.h"

static void freeCI(lua_State *L) {
  CallInfo *ci = L->ci;
  CallInfo *next = ci->next;
  ci->next = NULL;
  while ((ci = next) != NULL) {
    next = ci->next;
    luaM_free_(L, ci, sizeof(CallInfo));
    L->nci--;
  }
}
