#pragma once
#include "lobject.h"

typedef struct UpVal {
  GCObject *next;
  lu_byte tt;
  lu_byte marked;
  union {
    TValue *p;
    ptrdiff_t offset;
  } v;
  union {
    struct {
      struct UpVal *next;
      struct UpVal **previous;
    } open;
    TValue value;
  } u;
} UpVal;
