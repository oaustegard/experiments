#pragma once
#include "Value.h"
#include "lu_byte.h"

typedef struct TValue {
  Value value_;
  lu_byte tt_;
} TValue;
