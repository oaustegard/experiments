#pragma once
#include "LClosure.h"
#include "cclosure.h"

typedef union Closure {
  CClosure c;
  LClosure l;
} Closure;
