#pragma once
#include <lua.h>
#include "GCObject.h"

typedef union Value {
  GCObject *gc;
  void *p;
  lua_CFunction f;
  lua_Integer i;
  lua_Number n;
} Value;
