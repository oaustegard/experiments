#pragma once
#include "UValue.h"

struct Table;

typedef struct Udata {
  GCObject *next;
  lu_byte tt;
  lu_byte marked;
  unsigned short nuvalue;
  size_t len;
  struct Table *metatable;
  GCObject *gclist;
  UValue uv[1];
} Udata;
