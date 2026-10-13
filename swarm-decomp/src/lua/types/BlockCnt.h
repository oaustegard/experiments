#pragma once
#include "lobject.h"

typedef struct BlockCnt BlockCnt;
struct BlockCnt {
  BlockCnt *previous;
  int firstlabel;
  int firstgoto;
  lu_byte nactvar;
  lu_byte upval;
  lu_byte isloop;
  lu_byte insidetbc;
};
