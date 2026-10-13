#pragma once
#include "Proto.h"
#include "lu_byte.h"
#include "LexState.h"

struct BlockCnt;

typedef struct FuncState {
  Proto *f;
  struct FuncState *prev;
  LexState *ls;
  struct BlockCnt *bl;
  int pc;
  int lasttarget;
  int previousline;
  int nk;
  int np;
  int nabslineinfo;
  int firstlocal;
  int firstlabel;
  short ndebugvars;
  lu_byte nactvar;
  lu_byte nups;
  lu_byte freereg;
  lu_byte iwthabs;
  lu_byte needclose;
} FuncState;
