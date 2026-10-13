#include "LexState.h"
#include "FuncState.h"
#include "BlockCnt.h"
#include "enterblock.h"
#include "statlist.h"
#include "leaveblock.h"

static void block (LexState *ls) {
  FuncState *fs = ls->fs;
  BlockCnt bl;
  enterblock(fs, &bl, 0);
  statlist(ls);
  leaveblock(fs);
}
