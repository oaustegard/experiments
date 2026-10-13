#pragma once
#include "lobject.h"
#include "TString.h"

typedef union Vardesc {
  struct {
    Value value_;
    lu_byte tt_;
    lu_byte kind;
    lu_byte ridx;
    short pidx;
    TString *name;
  } vd;
  TValue k;
} Vardesc;
