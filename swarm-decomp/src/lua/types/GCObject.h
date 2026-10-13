#pragma once
#include "lu_byte.h"

typedef struct GCObject {
  struct GCObject *next;
  lu_byte tt;
  lu_byte marked;
} GCObject;
