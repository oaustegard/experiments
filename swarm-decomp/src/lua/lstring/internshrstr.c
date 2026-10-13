#include <limits.h>
#include <stddef.h>
#include <string.h>
#include "lstate.h"
#include "TString.h"
#include "fn_luaC_fullgc.h"
#include "fn_luaC_newobj.h"
#include "fn_luaD_throw.h"
#include "fn_luaS_hash.h"
#include "fn_luaS_resize.h"
#include "createstrobj.h"
#include "internshrstr.h"

#define WHITEBITS 0x18
#define l_unlikely(x) __builtin_expect(((x) != 0), 0)

static TString *internshrstr (lua_State *L, const char *str, size_t l) {
  TString *ts;
  global_State *g = G(L);
  stringtable *tb = &g->strt;
  unsigned int h = luaS_hash(str, l, g->seed);
  TString **list = &tb->hash[(int)(h & (tb->size - 1))];
  for (ts = *list; ts != NULL; ts = ts->u.hnext) {
    if (l == ts->shrlen && memcmp(str, ts->contents, l) == 0) {
      if (ts->marked & (g->currentwhite ^ WHITEBITS))
        ts->marked ^= WHITEBITS;
      return ts;
    }
  }
  if (l_unlikely(tb->nuse >= tb->size)) {
    if (l_unlikely(tb->nuse == INT_MAX)) {
      luaC_fullgc(L, 1);
      if (tb->nuse == INT_MAX)
        luaD_throw(L, LUA_ERRMEM);
    }
    if (tb->size <= INT_MAX / 2)
      luaS_resize(L, tb->size * 2);
    list = &tb->hash[(int)(h & (tb->size - 1))];
  }
  ts = createstrobj(L, l, LUA_TSTRING, h);
  ts->shrlen = (lu_byte)l;
  memcpy(ts->contents, str, l);
  ts->u.hnext = *list;
  *list = ts;
  tb->nuse++;
  return ts;
}
