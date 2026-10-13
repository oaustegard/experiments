#pragma once
#include "lobject.h"

/* A stack slot reference that is either a pointer or, while the stack is
   being reallocated, a byte offset from the stack base (same storage). */
typedef union StkIdRel {
  StkId p;
  ptrdiff_t offset;
} StkIdRel;
