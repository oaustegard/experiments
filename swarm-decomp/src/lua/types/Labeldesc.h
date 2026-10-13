#pragma once
#include "lobject.h"
#include "TString.h"

typedef struct Labeldesc {
  TString *name;
  int pc;
  int line;
  lu_byte nactvar;
  lu_byte close;
} Labeldesc;
