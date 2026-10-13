#include "FuncState.h"
#include "savelineinfo.h"
#include "lcode_limits.h"

void luaK_fixline (FuncState *fs, int line) {
  Proto *f = fs->f;
  int pc = fs->pc - 1;
  if (f->lineinfo[pc] != ABSLINEINFO) {
    fs->previousline -= f->lineinfo[pc];
    fs->iwthabs--;
  }
  else {
    fs->nabslineinfo--;
    fs->iwthabs = MAXIWTHABS + 1;
  }
  savelineinfo(fs, f, line);
}
