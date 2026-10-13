#include "lauxlib.h"

static void *resizebox (lua_State *L, int idx, size_t newsize) {
  void *ud;
  lua_Alloc allocf = lua_getallocf(L, &ud);
  UBox *box = (UBox *)lua_touserdata(L, idx);
  void *temp = allocf(ud, box->box, box->bsize, newsize);
  if (temp == NULL && newsize > 0) {
    lua_pushstring(L, "not enough memory");
    lua_error(L);
  }
  box->box = temp;
  box->bsize = newsize;
  return temp;
}
