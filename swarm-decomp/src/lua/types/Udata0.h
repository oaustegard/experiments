#pragma once
#include <stddef.h>
#include "Udata.h"

typedef struct Udata0 {
  GCObject *next;
  lu_byte tt;
  lu_byte marked;
  unsigned short nuvalue;
  size_t len;
  struct Table *metatable;
  char bindata[1];
} Udata0;
