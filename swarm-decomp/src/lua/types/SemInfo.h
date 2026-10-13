#pragma once
#include "lobject.h"
#include "TString.h"

typedef union SemInfo {
  lua_Number r;
  lua_Integer i;
  TString *ts;
} SemInfo;
