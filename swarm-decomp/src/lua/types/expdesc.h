#pragma once
#include <lua.h>
#include "expkind.h"
#include "lu_byte.h"

typedef struct expdesc {
  expkind k;
  union {
    lua_Integer ival;
    lua_Number nval;
    void *strval;
    int info;
    struct { short idx; lu_byte t; } ind;
    struct { lu_byte ridx; unsigned short vidx; } var;
  } u;
  int t;
  int f;
} expdesc;
