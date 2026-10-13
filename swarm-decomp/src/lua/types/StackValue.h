#pragma once
#include "TValue.h"

/* A stack slot: a TValue, or the to-be-closed list entry (same 16-byte size, so
   pointer arithmetic on StackValue * matches StkId arithmetic). */
typedef union StackValue {
  TValue val;
  struct {
    Value value_;
    lu_byte tt_;
    unsigned short delta;
  } tbclist;
} StackValue;
