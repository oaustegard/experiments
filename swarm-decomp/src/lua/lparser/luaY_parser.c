#include "LexState.h"
#include "FuncState.h"
#include "BlockCnt.h"
#include "Dyndata.h"
#include "LClosure.h"
#include "fn_luaF_newLclosure.h"
#include "fn_luaF_newproto.h"
#include "fn_luaD_inctop.h"
#include "fn_luaH_new.h"
#include "fn_luaS_new.h"
#include "fn_luaX_setinput.h"
#include "fn_luaX_next.h"
#include "fn_luaC_barrier_.h"
#include "fn_luaK_codeABCk.h"
#include "fn_luaY_parser.h"
#include "statement.h"
#include "open_func.h"
#include "close_func.h"
#include "error_expected.h"

LClosure *luaY_parser (lua_State *L, ZIO *z, Mbuffer *buff, Dyndata *dyd, const char *name, int firstchar) {
  LexState lexstate;
  FuncState funcstate;
  LClosure *cl = luaF_newLclosure(L, 1);
  L->top->value_.gc = (GCObject *)cl;
  L->top->tt_ = 0x46;
  luaD_inctop(L);
  lexstate.h = luaH_new(L);
  L->top->value_.gc = (GCObject *)lexstate.h;
  L->top->tt_ = 0x45;
  luaD_inctop(L);
  funcstate.f = cl->p = luaF_newproto(L);
  if ((((GCObject *)cl)->marked & 0x20) && (cl->p->marked & 0x18))
    luaC_barrier_(L, (GCObject *)cl, (GCObject *)cl->p);
  funcstate.f->source = luaS_new(L, name);
  if ((funcstate.f->marked & 0x20) && (funcstate.f->source->marked & 0x18))
    luaC_barrier_(L, (GCObject *)funcstate.f, (GCObject *)funcstate.f->source);
  lexstate.buff = buff;
  lexstate.dyd = dyd;
  dyd->actvar.n = dyd->gt.n = dyd->label.n = 0;
  luaX_setinput(L, &lexstate, z, funcstate.f->source, firstchar);
  {  /* mainfunc */
    BlockCnt bl;
    Upvaldesc *env;
    open_func(&lexstate, &funcstate, &bl);
    funcstate.f->is_vararg = 1;
    luaK_codeABCk(&funcstate, 0x51, 0, 0, 0, 0);
    env = allocupvalue(&funcstate);
    env->instack = 1;
    env->idx = 0;
    env->kind = 0;
    env->name = lexstate.envn;
    if ((funcstate.f->marked & 0x20) && (env->name->marked & 0x18))
      luaC_barrier_(lexstate.L, (GCObject *)funcstate.f, (GCObject *)env->name);
    luaX_next(&lexstate);
    statlist(&lexstate);
    if (lexstate.t.token != 0x120)
      error_expected(&lexstate, 0x120);
    close_func(&lexstate);
  }
  L->top--;
  return cl;
}
