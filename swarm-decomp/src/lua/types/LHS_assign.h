#pragma once
#include "expdesc.h"

struct LHS_assign {
  struct LHS_assign *prev;
  expdesc v;
};
