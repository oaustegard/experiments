#include "lstate.h"
#include "lua.h"
#include "Closure.h"
#include "fn_luaH_new.h"
#include "fn_luaH_setint.h"
#include "nextline.h"
#define noLuaClosure(f)	((f) == NULL || (f)->c.tt != 6)

static void collectvalidlines (lua_State *L, Closure *f) {
  if (noLuaClosure(f)) {
    L->top->tt_ = 0;  /* setnilvalue */
    L->top++;
  }
  else {
    int i;
    TValue v;
    const Proto *p = f->l.p;
    int currline = p->linedefined;
    Table *t = luaH_new(L);  /* new table to store active lines */
    L->top->tt_ = 0x45;
    L->top->value_.gc = (GCObject *)t;  /* push it on stack */
    L->top++;
    if (p->lineinfo != NULL) {
      v.tt_ = 0x11;  /* setbtvalue: boolean 'true' for all indices */
      if (!p->is_vararg)  /* regular function? */
        i = 0;  /* consider all instructions */
      else {  /* vararg function */
        currline = nextline(p, currline, 0);  /* skip first instruction */
        i = 1;
      }
      for (; i < p->sizelineinfo; i++) {  /* for each instruction */
        currline = nextline(p, currline, i);  /* get its line */
        luaH_setint(L, t, currline, &v);  /* table[line] = true */
      }
    }
  }
}
