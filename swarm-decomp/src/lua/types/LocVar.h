#pragma once

struct TString;

typedef struct LocVar {
  struct TString *varname;
  int startpc;
  int endpc;
} LocVar;
