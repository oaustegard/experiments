#pragma once
#include <stddef.h>
#include "lobject.h"
#include "Node.h"

typedef struct Table {
  GCObject *next;
  lu_byte tt;
  lu_byte marked;
  lu_byte flags;
  lu_byte lsizenode;
  unsigned int alimit;
  TValue *array;
  Node *node;
  Node *lastfree;
  struct Table *metatable;
  GCObject *gclist;
} Table;
