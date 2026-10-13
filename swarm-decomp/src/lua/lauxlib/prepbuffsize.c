#include <string.h>
#include "lauxlib.h"
#include "newbuffsize.h"
#include "resizebox.h"
#include "boxgc.h"
#include "prepbuffsize.h"

static char *prepbuffsize (luaL_Buffer *B, size_t sz, int boxidx) {
  if (B->size - B->n >= sz)
    return B->b + B->n;
  else {
    lua_State *L = B->L;
    char *newbuff;
    size_t newsize = newbuffsize(B, sz);
    if (B->b != B->init.b)
      newbuff = (char *)resizebox(L, boxidx, newsize);
    else {
      UBox *box;
      lua_remove(L, boxidx);
      box = (UBox *)lua_newuserdatauv(L, sizeof(UBox), 0);
      box->box = NULL;
      box->bsize = 0;
      if (luaL_newmetatable(L, "_UBOX*"))
        luaL_setfuncs(L, boxmt, 0);
      lua_setmetatable(L, -2);
      lua_insert(L, boxidx);
      lua_toclose(L, boxidx);
      newbuff = (char *)resizebox(L, boxidx, newsize);
      memcpy(newbuff, B->b, B->n * sizeof(char));
    }
    B->b = newbuff;
    B->size = newsize;
    return newbuff + B->n;
  }
}
