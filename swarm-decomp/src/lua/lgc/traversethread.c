#include "traversethread.h"
#include "fn_luaD_shrinkstack.h"
#define LINKGRAY(o, pnext, list) do { GCObject *lg_o_ = (GCObject *)(o); *(pnext) = *(list); *(list) = lg_o_; set2gray(lg_o_); } while (0)
#include "reallymarkobject.h"

#define EXTRA_STACK 5

static int traversethread(global_State *g, lua_State *th) {
  UpVal *uv;
  StkId o = th->stack;
  if (isold(th) || g->gcstate == GCSpropagate)
    LINKGRAY((GCObject *)th, &th->gclist, &g->grayagain);
  if (o == NULL)
    return 1;
  for (; o < th->top; o++)
    markvalue(g, s2v(o));
  for (uv = th->openupval; uv != NULL; uv = uv->u.open.next)
    markobject(g, uv);
  if (g->gcstate == GCSatomic) {
    if (!g->gcemergency)
      luaD_shrinkstack(th);
    for (o = th->top; o < th->stack_last + EXTRA_STACK; o++)
      setnilvalue(s2v(o));
    if (th->twups == th && th->openupval != NULL) {
      th->twups = g->twups;
      g->twups = th;
    }
  }
  return 1 + (int)(th->stack_last - th->stack);
}
