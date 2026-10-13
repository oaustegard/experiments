#include <limits.h>
#include "Table.h"

static Node *mainpositionTV (const Table *t, const TValue *key) {
  Value kv = key->value_;
  const Value *kvl = &kv;
  switch (key->tt_ & 0x3f) {
    case 0x03: {  /* LUA_VNUMINT */
      lua_Integer i = kvl->i;
      return hashint(t, i);
    }
    case 0x13:  /* LUA_VNUMFLT */
      return &t->node[l_hashfloat(kvl->n) % (((1 << t->lsizenode) - 1) | 1)];
    case 0x04: {  /* LUA_VSHRSTR */
      TString *ts = (TString *)kvl->gc;
      return &t->node[(int)(ts->hash & ((1 << t->lsizenode) - 1))];
    }
    case 0x14: {  /* LUA_VLNGSTR */
      TString *ts = (TString *)kvl->gc;
      return &t->node[(int)(luaS_hashlongstr(ts) & ((1 << t->lsizenode) - 1))];
    }
    case 0x01:  /* LUA_VFALSE */
      return &t->node[0];
    case 0x11:  /* LUA_VTRUE */
      return &t->node[(int)(1 & ((1 << t->lsizenode) - 1))];
    case 0x02: {  /* LUA_VLIGHTUSERDATA */
      unsigned int u = (unsigned int)((size_t)kvl->p & UINT_MAX);
      return &t->node[u % (((1 << t->lsizenode) - 1) | 1)];
    }
    case 0x16: {  /* LUA_VLCF */
      unsigned int u = (unsigned int)((size_t)(void *)kvl->f & UINT_MAX);
      return &t->node[u % (((1 << t->lsizenode) - 1) | 1)];
    }
    default: {
      unsigned int u = (unsigned int)((size_t)kvl->gc & UINT_MAX);
      return &t->node[u % (((1 << t->lsizenode) - 1) | 1)];
    }
  }
}
