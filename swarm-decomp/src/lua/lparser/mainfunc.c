#include "LexState.h"
#include "FuncState.h"
#include "BlockCnt.h"
#include "Upvaldesc.h"
#include "fn_luaX_next.h"
#include "luaC_objbarrier.h"
#include "open_func.h"
#include "close_func.h"
#include "allocupvalue.h"
#include "check.h"
#include "block_follow.h"
#include "statlist.h"
#include "setvararg.h"

static void mainfunc (LexState *ls, FuncState *fs) {
  BlockCnt bl;
  Upvaldesc *env;
  open_func(ls, fs, &bl);
  setvararg(fs, 0);
  env = allocupvalue(fs);
  env->instack = 1;
  env->idx = 0;
  env->kind = 0;
  env->name = ls->envn;
  luaC_objbarrier(ls->L, fs->f, env->name);
  luaX_next(ls);
  statlist(ls);
  check(ls, 0x120);
  close_func(ls);
}
