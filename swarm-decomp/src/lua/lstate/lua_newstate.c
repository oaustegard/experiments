#include "lua.h"
#include "lstate.h"
#include "fn_luaD_rawrunprotected.h"
#include "fn_luaM_malloc_.h"

static void f_luaopen (lua_State *L, void *ud);
static void close_state (lua_State *L);

lua_State *lua_newstate (lua_Alloc f, void *ud) {
  int i;
  lua_State *L;
  global_State *g;
  LG *l = (LG *)(*f)(ud, NULL, 8, sizeof(LG));
  if (l == NULL) return NULL;
  L = &l->l.l;
  g = &l->g;
  L->tt = 8;
  g->currentwhite = 8;
  L->marked = g->currentwhite & 0x18;
  L->l_G = g;
  L->stack = NULL;
  L->ci = NULL;
  L->nci = 0;
  L->twups = L;
  L->nCcalls = 0;
  L->errorJmp = NULL;
  L->hook = NULL;
  L->hookmask = 0;
  L->basehookcount = 0;
  L->allowhook = 1;
  L->hookcount = 0;
  L->openupval = NULL;
  L->status = 0;
  L->errfunc = 0;
  L->oldpc = 0;
  g->allgc = (GCObject *)L;
  L->next = NULL;
  L->nCcalls += 0x10000;
  g->frealloc = f;
  g->ud = ud;
  g->warnf = NULL;
  g->ud_warn = NULL;
  g->mainthread = L;
  g->seed = luai_makeseed(L);
  g->gcstp = 2;
  g->strt.nuse = 0;
  g->strt.size = 0;
  g->strt.hash = NULL;
  g->l_registry.tt_ = 0;
  g->panic = NULL;
  g->gcstate = 8;
  g->gckind = 0;
  g->gcstopem = 0;
  g->gcemergency = 0;
  g->sweepgc = NULL;
  g->finobj = g->tobefnz = g->fixedgc = NULL;
  g->firstold1 = g->survival = g->reallyold = g->old1 = NULL;
  g->finobjsur = g->finobjold1 = g->finobjrold = NULL;
  g->gray = g->grayagain = NULL;
  g->weak = g->ephemeron = g->allweak = NULL;
  g->twups = NULL;
  g->totalbytes = sizeof(LG);
  g->GCdebt = 0;
  g->lastatomic = 0;
  g->nilvalue.value_.i = 0;
  g->nilvalue.tt_ = 3;
  g->gcpause = 200 / 4;
  g->gcstepmul = 100 / 4;
  g->gcstepsize = 13;
  g->genmajormul = 100 / 4;
  g->genminormul = 20;
  for (i = 0; i < 9; i++) g->mt[i] = NULL;
  if (luaD_rawrunprotected(L, f_luaopen, NULL) != 0) {
    close_state(L);
    L = NULL;
  }
  return L;
}
