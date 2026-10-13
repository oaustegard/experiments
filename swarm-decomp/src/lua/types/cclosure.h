#pragma once
#include "lobject.h"

typedef struct CClosure {
  GCObject *next;
  lu_byte tt;
  lu_byte marked;
  lu_byte nupvalues;
  GCObject *gclist;
  lua_CFunction f;
  TValue upvalue[1];
} CClosure;
