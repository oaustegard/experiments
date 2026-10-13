#include "FuncState.h"
#include "BlockCnt.h"
#include "Upvaldesc.h"
#include "Dyndata.h"
#include "fn_luaC_barrier_.h"
#include "fn_luaK_codeABx.h"
#include "fn_luaK_codeABCk.h"
#include "fn_luaK_exp2nextreg.h"
#include "fn_luaK_reserveregs.h"
#include "fn_luaX_next.h"
#include "fn_luaX_newstring.h"
#include "fn_luaX_syntaxerror.h"
#include "luaC_objbarrier.h"
#include "addprototype.h"
#include "checknext.h"
#include "str_checkname.h"
#include "open_func.h"
#include "close_func.h"
#include "statement.h"
#include "check_match.h"
#include "error_expected.h"
#include "new_localvar.h"
#include "adjustlocalvars.h"
#include "testnext.h"

static void body (LexState *ls, expdesc *e, int ismethod, int line) {
  FuncState new_fs;
  BlockCnt bl;
  new_fs.f = addprototype(ls);
  new_fs.f->linedefined = line;
  open_func(ls, &new_fs, &bl);
  checknext(ls, '(');
  if (ismethod) {
    new_localvar(ls, luaX_newstring(ls, "self", 4));
    adjustlocalvars(ls, 1);
  }
  {
    FuncState *fs = ls->fs;
    Proto *f = fs->f;
    int nparams = 0;
    int isvararg = 0;
    if (ls->t.token != ')') {
      do {
        switch (ls->t.token) {
          case 0x123: {
            new_localvar(ls, str_checkname(ls));
            nparams++;
            break;
          }
          case 0x118: {
            luaX_next(ls);
            isvararg = 1;
            break;
          }
          default: luaX_syntaxerror(ls, "<name> or '...' expected");
        }
      } while (!isvararg && testnext(ls, ','));
    }
    adjustlocalvars(ls, nparams);
    f->numparams = fs->nactvar;
    if (isvararg) {
      fs->f->is_vararg = 1;
      luaK_codeABCk(fs, 0x51, f->numparams, 0, 0, 0);
    }
    luaK_reserveregs(fs, fs->nactvar);
  }
  checknext(ls, ')');
  statlist(ls);
  new_fs.f->lastlinedefined = ls->linenumber;
  check_match(ls, 0x105, 0x108, line);
  {
    FuncState *fs = ls->fs->prev;
    int pc = luaK_codeABx(fs, 79, 0, fs->np - 1);
    e->f = e->t = -1;
    e->k = VRELOC;
    e->u.info = pc;
    luaK_exp2nextreg(fs, e);
  }
  close_func(ls);
}
