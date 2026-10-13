#pragma once
#include "rand64.h"
typedef struct RanState {
  Rand64 s[4];
} RanState;
