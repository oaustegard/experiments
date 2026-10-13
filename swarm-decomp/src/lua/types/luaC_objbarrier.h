#pragma once
#include "fn_luaC_barrier_.h"
#include "lgc_defs.h"

#define luaC_objbarrier(L,p,o) \
  ((isblack(p) && iswhite(o)) ? (luaC_barrier_(L, (GCObject *)(p), (GCObject *)(o))) : (void)0)
