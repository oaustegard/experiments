#include "lstate.h"
#include "cclosure.h"
#include "LClosure.h"
#include "UpVal.h"
#include "Proto.h"
#include "lua.h"

#define LUA_VCCL (6 | (2 << 4))
#define getstr(ts) ((ts)->contents)
#define obj2gco(v) ((GCObject *)(v))

static const char *aux_upvalue (TValue *fi, int n, TValue **val, GCObject **owner) {
  switch (ttypetag(fi)) {
    case LUA_VCCL: {
      CClosure *f = (CClosure *)fi->value_.gc;
      if (!((unsigned int)n - 1u < (unsigned int)f->nupvalues))
        return NULL;
      *val = &f->upvalue[n-1];
      if (owner) *owner = obj2gco(f);
      return "";
    }
    case LUA_VLCL: {
      LClosure *f = (LClosure *)fi->value_.gc;
      TString *name;
      Proto *p = f->p;
      if (!((unsigned int)n - 1u < (unsigned int)p->sizeupvalues))
        return NULL;
      *val = f->upvals[n-1]->v.p;
      if (owner) *owner = obj2gco(f->upvals[n - 1]);
      name = p->upvalues[n-1].name;
      return (name == NULL) ? "(no name)" : getstr(name);
    }
    default: return NULL;
  }
}
