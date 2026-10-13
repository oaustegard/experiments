#pragma once
#include "lobject.h"
#include "TString.h"

typedef struct Upvaldesc {
  TString *name;
  lu_byte instack;
  lu_byte idx;
  lu_byte kind;
} Upvaldesc;
