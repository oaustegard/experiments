#include <stdlib.h>
#include "savelineinfo.h"
#include "lcode_limits.h"
#include "fn_luaM_growaux_.h"

static void savelineinfo (FuncState *fs, Proto *f, int line) {
  int linedif = line - fs->previousline;
  int pc = fs->pc - 1;
  if (abs(linedif) >= LIMLINEDIFF || fs->iwthabs++ >= MAXIWTHABS) {
    f->abslineinfo = luaM_growaux_(fs->ls->L, f->abslineinfo, fs->nabslineinfo,
                                   &f->sizeabslineinfo, sizeof(AbsLineInfo), MAX_INT, "lines");
    f->abslineinfo[fs->nabslineinfo].pc = pc;
    f->abslineinfo[fs->nabslineinfo++].line = line;
    linedif = ABSLINEINFO;
    fs->iwthabs = 1;
  }
  f->lineinfo = luaM_growaux_(fs->ls->L, f->lineinfo, pc, &f->sizelineinfo,
                              sizeof(ls_byte), MAX_INT, "opcodes");
  f->lineinfo[pc] = linedif;
  fs->previousline = line;
}
