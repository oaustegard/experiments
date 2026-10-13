#include "LexState.h"
#include "FuncState.h"
#include "BlockCnt.h"
#include "Dyndata.h"
#include "LHS_assign.h"
#include "fn_luaE_incCstack.h"
#include "fn_luaX_next.h"
#include "fn_luaS_newlstr.h"
#include "fn_luaO_pushfstring.h"
#include "fn_luaX_syntaxerror.h"
#include "fn_luaK_jump.h"
#include "luaK_getlabel.h"
#include "fn_luaK_patchlist.h"
#include "fn_luaK_patchtohere.h"
#include "fn_luaK_semerror.h"
#include "fn_luaK_storevar.h"
#include "fn_luaK_setreturns.h"
#include "fn_luaK_ret.h"
#include "fn_luaK_exp2anyreg.h"
#include "fn_luaK_exp2nextreg.h"
#include "fn_luaK_exp2anyregup.h"
#include "fn_luaK_exp2const.h"
#include "fn_luaK_indexed.h"
#include "fn_luaK_goiftrue.h"
#include "fn_luaK_fixline.h"
#include "fn_luaK_checkstack.h"
#include "fn_luaK_reserveregs.h"
#include "fn_luaK_int.h"
#include "fn_luaY_nvarstack.h"
#include "fn_luaK_codeABCk.h"
#include "fn_luaX_newstring.h"
#include "statement.h"
#include "block.h"
#include "forbody.h"
#include "body.h"
#include "test_then_block.h"
#include "restassign.h"
#include "suffixedexp.h"
#include "check_readonly.h"
#include "adjust_assign.h"
#include "adjustlocalvars.h"
#include "createlabel.h"
#include "leaveblock.h"
#include "fieldsel.h"
#include "singlevar.h"
#include "subexpr.h"
#include "check_match.h"
#include "error_expected.h"
#include "new_localvar.h"
#include "newlabelentry.h"
#include "LocVar.h"
#include "reglevel.h"
#include "labelstat.h"
#include "testnext.h"
#include <string.h>

static void statement (LexState *ls) {
  int line = ls->linenumber;
  luaE_incCstack(ls->L);
  switch (ls->t.token) {
    case ';': {
      luaX_next(ls);
      break;
    }
    case 0x10a: {  /* IF */
      FuncState *fs = ls->fs;
      int escapelist = -1;
      test_then_block(ls, &escapelist);
      while (ls->t.token == 0x104)
        test_then_block(ls, &escapelist);
      if (ls->t.token == 0x103) {
        luaX_next(ls);
        block(ls);
      }
      check_match(ls, 0x105, 0x10a, line);
      luaK_patchtohere(fs, escapelist);
      break;
    }
    case 0x101: {  /* BREAK */
      int pc;
      line = ls->linenumber;
      luaX_next(ls);
      pc = luaK_jump(ls->fs);
      newlabelentry(ls, &ls->dyd->gt, luaS_newlstr(ls->L, "break", 5), line, pc);
      break;
    }
    case 0x102: {  /* DO */
      luaX_next(ls);
      block(ls);
      check_match(ls, 0x105, 0x102, line);
      break;
    }
    case 0x107: {  /* FOR */
      FuncState *fs = ls->fs;
      BlockCnt bl;
      TString *varname;
      int base;
      enterblock(fs, &bl, 1);
      luaX_next(ls);
      if (ls->t.token != 0x123)
        error_expected(ls, 0x123);
      varname = ls->t.seminfo.ts;
      luaX_next(ls);
      switch (ls->t.token) {
        case '=': {  /* fornum */
          FuncState *fs = ls->fs;
          expdesc e;
          base = fs->freereg;
          new_localvar(ls, luaX_newstring(ls, "(for state)", 11));
          new_localvar(ls, luaX_newstring(ls, "(for state)", 11));
          new_localvar(ls, luaX_newstring(ls, "(for state)", 11));
          new_localvar(ls, varname);
          if (ls->t.token != '=')
            error_expected(ls, '=');
          luaX_next(ls);
          subexpr(ls, &e, 0);
          luaK_exp2nextreg(ls->fs, &e);
          if (ls->t.token != ',')
            error_expected(ls, ',');
          luaX_next(ls);
          subexpr(ls, &e, 0);
          luaK_exp2nextreg(ls->fs, &e);
          if (ls->t.token == ',') {
            luaX_next(ls);
            subexpr(ls, &e, 0);
            luaK_exp2nextreg(ls->fs, &e);
          }
          else {
            luaK_int(fs, fs->freereg, 1);
            luaK_reserveregs(fs, 1);
          }
          adjustlocalvars(ls, 3);
          forbody(ls, base, line, 1, 0);
          break;
        }
        case ',': case 0x10b: {  /* forlist */
          FuncState *fs = ls->fs;
          expdesc e;
          int nvars = 5;
          int line;
          int n;
          base = fs->freereg;
          new_localvar(ls, luaX_newstring(ls, "(for state)", 11));
          new_localvar(ls, luaX_newstring(ls, "(for state)", 11));
          new_localvar(ls, luaX_newstring(ls, "(for state)", 11));
          new_localvar(ls, luaX_newstring(ls, "(for state)", 11));
          new_localvar(ls, varname);
          while (testnext(ls, ',')) {
            TString *name;
            if (ls->t.token != 0x123)
              error_expected(ls, 0x123);
            name = ls->t.seminfo.ts;
            nvars++;
            luaX_next(ls);
            new_localvar(ls, name);
          }
          if (ls->t.token != 0x10b)
            error_expected(ls, 0x10b);
          luaX_next(ls);
          line = ls->linenumber;
          n = 1;
          subexpr(ls, &e, 0);
          while (testnext(ls, ',')) {
            luaK_exp2nextreg(ls->fs, &e);
            subexpr(ls, &e, 0);
            n++;
          }
          adjust_assign(ls, 4, n, &e);
          adjustlocalvars(ls, 4);
          fs->bl->upval = 1;
          fs->bl->insidetbc = 1;
          fs->needclose = 1;
          luaK_checkstack(fs, 3);
          forbody(ls, base, line, nvars - 4, 1);
          break;
        }
        default:
          luaX_syntaxerror(ls, "'=' or 'in' expected");
      }
      check_match(ls, 0x105, 0x107, line);
      leaveblock(fs);
      break;
    }
    case 0x108: {  /* FUNCTION */
      expdesc v, b;
      int ismethod;
      luaX_next(ls);
      ismethod = 0;
      singlevar(ls, &v);
      while (ls->t.token == '.') {
        FuncState *fs = ls->fs;
        expdesc key;
        luaK_exp2anyregup(fs, &v);
        luaX_next(ls);
        if (ls->t.token != 0x123)
          error_expected(ls, 0x123);
        {
          TString *name = ls->t.seminfo.ts;
          luaX_next(ls);
          key.f = key.t = -1;
          key.u.strval = name;
          key.k = VKSTR;
        }
        luaK_indexed(fs, &v, &key);
      }
      if (ls->t.token == ':') {
        ismethod = 1;
        fieldsel(ls, &v);
      }
      body(ls, &b, ismethod, line);
      check_readonly(ls, &v);
      luaK_storevar(ls->fs, &v, &b);
      luaK_fixline(ls->fs, line);
      break;
    }
    case 0x109: {  /* GOTO */
      FuncState *fs;
      int line;
      TString *name;
      Labeldesc *lb = NULL;
      int i;
      luaX_next(ls);
      fs = ls->fs;
      line = ls->linenumber;
      if (ls->t.token != 0x123)
        error_expected(ls, 0x123);
      name = ls->t.seminfo.ts;
      luaX_next(ls);
      {
        Dyndata *dyd = ls->dyd;
        for (i = ls->fs->firstlabel; i < dyd->label.n; i++) {
          if (dyd->label.arr[i].name == name) {
            lb = &dyd->label.arr[i];
            break;
          }
        }
      }
      if (lb == NULL) {
        newlabelentry(ls, &ls->dyd->gt, name, line, luaK_jump(fs));
      }
      else {
        int lblevel = reglevel(fs, lb->nactvar);
        if (luaY_nvarstack(fs) > lblevel)
          luaK_codeABCk(fs, 0x36, lblevel, 0, 0, 0);
        luaK_patchlist(fs, luaK_jump(fs), lb->pc);
      }
      break;
    }
    case 0x10c: {  /* LOCAL */
      FuncState *fs;
      int toclose = -1;
      int vidx = 0;
      int nvars = 0;
      int nexps;
      expdesc e;
      luaX_next(ls);
      fs = ls->fs;
      if (ls->t.token == 0x108) {  /* localfunc */
        expdesc b;
        int fvar;
        luaX_next(ls);
        fvar = fs->nactvar;
        if (ls->t.token != 0x123)
          error_expected(ls, 0x123);
        {
          TString *name = ls->t.seminfo.ts;
          luaX_next(ls);
          new_localvar(ls, name);
        }
        adjustlocalvars(ls, 1);
        body(ls, &b, 0, ls->linenumber);
        {
          Vardesc *vd = &ls->dyd->actvar.arr[fs->firstlocal + fvar];
          if (vd->vd.kind != 3) {
            LocVar *lv = &fs->f->locvars[vd->vd.pidx];
            lv->startpc = fs->pc;
          }
        }
        break;
      }
      /* localstat */
      do {
        int kind = 0;
        TString *name;
        if (ls->t.token != 0x123)
          error_expected(ls, 0x123);
        name = ls->t.seminfo.ts;
        luaX_next(ls);
        vidx = new_localvar(ls, name);
        if (ls->t.token == '<') {  /* getlocalattribute */
          const char *attr;
          luaX_next(ls);
          if (ls->t.token != 0x123)
            error_expected(ls, 0x123);
          attr = ls->t.seminfo.ts->contents;
          luaX_next(ls);
          if (ls->t.token != '>')
            error_expected(ls, '>');
          luaX_next(ls);
          if (strcmp(attr, "const") == 0)
            kind = 1;
          else if (strcmp(attr, "close") == 0)
            kind = 2;
          else
            luaK_semerror(ls, luaO_pushfstring(ls->L, "unknown attribute '%s'", attr));
        }
        ls->dyd->actvar.arr[fs->firstlocal + vidx].vd.kind = kind;
        if (kind == 2) {
          if (toclose != -1)
            luaK_semerror(ls, "multiple to-be-closed variables in local list");
          toclose = fs->nactvar + nvars;
        }
        nvars++;
      } while (ls->t.token == ',' && (luaX_next(ls), 1));
      if (ls->t.token == '=') {
        luaX_next(ls);
        nexps = 1;
        subexpr(ls, &e, 0);
        while (ls->t.token == ',') {
          luaK_exp2nextreg(fs, &e);
          luaX_next(ls);
          subexpr(ls, &e, 0);
          nexps++;
        }
      }
      else {
        e.k = VVOID;
        nexps = 0;
      }
      {
        Vardesc *var = &ls->dyd->actvar.arr[fs->firstlocal + vidx];
        if (nvars == nexps && var->vd.kind == 1 &&
            luaK_exp2const(fs, &e, &var->k)) {
          var->vd.kind = 3;
          adjustlocalvars(ls, nvars - 1);
          fs->nactvar++;
        }
        else {
          adjust_assign(ls, nvars, nexps, &e);
          adjustlocalvars(ls, nvars);
        }
      }
      if (toclose != -1) {
        fs->bl->upval = 1;
        fs->bl->insidetbc = 1;
        fs->needclose = 1;
        {
          int level = toclose;
          int nv = level;
          int reg = 0;
          while (nv-- > 0) {
            Vardesc *vd = &ls->dyd->actvar.arr[fs->firstlocal + nv];
            if (vd->vd.kind != 3) {
              reg = vd->vd.ridx + 1;
              break;
            }
          }
          luaK_codeABCk(fs, 0x37, reg, 0, 0, 0);
        }
      }
      break;
    }
    case 0x110: {  /* REPEAT */
      FuncState *fs = ls->fs;
      int repeat_init = luaK_getlabel(fs);
      BlockCnt bl1, bl2;
      int condexit;
      enterblock(fs, &bl1, 1);
      enterblock(fs, &bl2, 0);
      luaX_next(ls);
      while (!block_follow(ls, 1)) {
        if (ls->t.token == 0x111) {
          statement(ls);
          break;
        }
        statement(ls);
      }
      check_match(ls, 0x114, 0x110, line);
      {
        expdesc v;
        subexpr(ls, &v, 0);
        if (v.k == VNIL)
          v.k = VFALSE;
        luaK_goiftrue(ls->fs, &v);
        condexit = v.f;
      }
      leaveblock(fs);
      if (bl2.upval) {
        int exit = luaK_jump(fs);
        int nvar = bl2.nactvar;
        int reg = 0;
        luaK_patchtohere(fs, condexit);
        while (nvar-- > 0) {
          Vardesc *vd = &ls->dyd->actvar.arr[fs->firstlocal + nvar];
          if (vd->vd.kind != 3) {
            reg = vd->vd.ridx + 1;
            break;
          }
        }
        luaK_codeABCk(fs, 0x36, reg, 0, 0, 0);
        condexit = luaK_jump(fs);
        luaK_patchtohere(fs, exit);
      }
      luaK_patchlist(fs, condexit, repeat_init);
      leaveblock(fs);
      break;
    }
    case 0x111: {  /* RETURN */
      luaX_next(ls);
      retstat(ls);
      break;
    }
    case 0x115: {  /* WHILE */
      whilestat(ls, line);
      break;
    }
    case 0x11f: {  /* DBCOLON: label */
      TString *name;
      luaX_next(ls);
      if (ls->t.token != 0x123)
        error_expected(ls, 0x123);
      name = ls->t.seminfo.ts;
      luaX_next(ls);
      labelstat(ls, name, line);
      break;
    }
    default: {  /* exprstat */
      struct LHS_assign v;
      FuncState *fs = ls->fs;
      suffixedexp(ls, &v.v);
      if (ls->t.token == '=' || ls->t.token == ',') {
        v.prev = NULL;
        restassign(ls, &v, 1);
      }
      else {
        if (v.v.k != VCALL)
          luaX_syntaxerror(ls, "syntax error");
        ((lu_byte *)&fs->f->code[v.v.u.info])[3] = 1;
      }
      break;
    }
  }
  ls->fs->freereg = luaY_nvarstack(ls->fs);
  ls->L->nCcalls--;
}
