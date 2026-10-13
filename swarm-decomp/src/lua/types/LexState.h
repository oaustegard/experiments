#pragma once
#include "Zio.h"
#include "Mbuffer.h"
#include "Token.h"
#include "TString.h"

typedef struct LexState {
  int current;
  int linenumber;
  int lastline;
  Token t;
  Token lookahead;
  struct FuncState *fs;
  struct lua_State *L;
  ZIO *z;
  Mbuffer *buff;
  Table *h;
  struct Dyndata *dyd;
  TString *source;
  TString *envn;
} LexState;
