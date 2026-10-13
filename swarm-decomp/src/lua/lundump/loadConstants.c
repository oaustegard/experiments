#include "lundump.h"

static void loadConstants (LoadState *S, Proto *f) {
  int i;
  int n = loadInt(S);
  f->k = luaM_malloc_(S->L, n * sizeof(TValue), 0);
  f->sizek = n;
  for (i = 0; i < n; i++)
    setnilvalue(&f->k[i]);
  for (i = 0; i < n; i++) {
    TValue *o = &f->k[i];
    int t = loadByte(S);
    switch (t) {
      case 0x00:  /* LUA_VNIL */
        setnilvalue(o);
        break;
      case 0x01:  /* LUA_VFALSE */
        o->tt_ = 0x01;
        break;
      case 0x11:  /* LUA_VTRUE */
        o->tt_ = 0x11;
        break;
      case 0x13: {  /* LUA_VNUMFLT */
        lua_Number x;
        loadBlock(S, &x, sizeof(x));
        o->value_.n = x;
        o->tt_ = 0x13;
        break;
      }
      case 0x03: {  /* LUA_VNUMINT */
        lua_Integer x;
        loadBlock(S, &x, sizeof(x));
        o->value_.i = x;
        o->tt_ = 0x03;
        break;
      }
      case 0x04:  /* LUA_VSHRSTR */
      case 0x14:  /* LUA_VLNGSTR */
        o->value_.gc = (GCObject *)loadString(S, f);
        o->tt_ = o->value_.gc->tt | 0x40;
        break;
      default:
        break;
    }
  }
}
