#include "addk.h"
#include "lcode_limits.h"
#include "fn_luaH_get.h"
#include "fn_luaH_finishset.h"
#include "fn_luaM_growaux_.h"
#include "fn_luaV_equalobj.h"
#include "fn_luaC_barrier_.h"
#include "lobject_tags.h"
#include "lopcodes.h"

static int addk (FuncState *fs, TValue *key, TValue *v) {
  TValue val;
  lua_State *L = fs->ls->L;
  Proto *f = fs->f;
  const TValue *idx = luaH_get(fs->ls->h, key);
  int k, oldsize;
  if (ttisinteger(idx)) {
    k = (int)ivalue(idx);
    if (k < fs->nk && ttypetag(&f->k[k]) == ttypetag(v) &&
                      luaV_equalobj(NULL, &f->k[k], v))
      return k;
  }
  oldsize = f->sizek;
  k = fs->nk;
  setivalue(&val, k);
  luaH_finishset(L, fs->ls->h, key, idx, &val);
  f->k = luaM_growaux_(L, f->k, k, &f->sizek, sizeof(TValue), 0x1ffffff, "constants");
  while (oldsize < f->sizek) setnilvalue(&f->k[oldsize++]);
  setobj(&f->k[k], v);
  fs->nk++;
  if ((v->tt_ & BIT_ISCOLLECTABLE) && (f->marked & (1 << BLACKBIT)) &&
      (v->value_.gc->marked & WHITEBITS))
    luaC_barrier_(L, (GCObject *)f, v->value_.gc);
  return k;
}
