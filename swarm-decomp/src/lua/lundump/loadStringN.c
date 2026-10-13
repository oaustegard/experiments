#include "lundump.h"

static TString *loadStringN (LoadState *S, Proto *p) {
  lua_State *L = S->L;
  TString *ts;
  size_t size = loadSize(S);
  if (size == 0)  /* no string? */
    return NULL;
  else if (--size <= 40) {  /* short string? */
    char buff[40];
    loadVector(S, buff, size);  /* load string into buffer */
    ts = luaS_newlstr(L, buff, size);  /* create string */
  }
  else {  /* long string */
    ts = luaS_createlngstrobj(L, size);  /* create string */
    L->top->value_.gc = (GCObject *)ts;  /* anchor it ('loadVector' can GC) */
    L->top->tt_ = ts->tt | 0x40;
    luaD_inctop(L);
    loadVector(S, ts->contents, size);  /* load directly in final place */
    L->top--;  /* pop string */
  }
  if (isblack(p) && iswhite(ts))
    luaC_barrier_(L, (GCObject *)p, (GCObject *)ts);
  return ts;
}
