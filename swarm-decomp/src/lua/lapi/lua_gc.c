#include <stdarg.h>
#include "lstate.h"
#include "lgc_defs.h"
#include "lua.h"
#include "fn_luaE_setdebt.h"
#include "fn_luaC_fullgc.h"
#include "fn_luaC_step.h"
#include "fn_luaC_changemode.h"

#define GCSTPGC 2
#define GCSTPUSR 1
#define getgcparam(p) ((p) * 4)
#define setgcparam(p,v) ((p) = (v) / 4)
#define gettotalbytes(g) ((lu_mem)((g)->totalbytes + (g)->GCdebt))
#define gcrunning(g) ((g)->gcstp == 0)
#define isdecGCmodegen(g) ((g)->gckind == KGC_GEN || (g)->lastatomic != 0)
#define cast_int(i) ((int)(i))
#define cast(t, exp) ((t)(exp))

LUA_API int lua_gc (lua_State *L, int what, ...) {
  va_list argp;
  int res = 0;
  global_State *g = L->l_G;
  if (g->gcstp & GCSTPGC)
    return -1;
  va_start(argp, what);
  switch (what) {
    case LUA_GCSTOP: {
      g->gcstp = GCSTPUSR;
      break;
    }
    case LUA_GCRESTART: {
      luaE_setdebt(g, 0);
      g->gcstp = 0;
      break;
    }
    case LUA_GCCOLLECT: {
      luaC_fullgc(L, 0);
      break;
    }
    case LUA_GCCOUNT: {
      res = cast_int(gettotalbytes(g) >> 10);
      break;
    }
    case LUA_GCCOUNTB: {
      res = cast_int(gettotalbytes(g) & 0x3ff);
      break;
    }
    case LUA_GCSTEP: {
      int data = va_arg(argp, int);
      l_mem debt = 1;
      lu_byte oldstp = g->gcstp;
      g->gcstp = 0;
      if (data == 0) {
        luaE_setdebt(g, 0);
        luaC_step(L);
      }
      else {
        debt = cast(l_mem, data) * 1024 + g->GCdebt;
        luaE_setdebt(g, debt);
        if (L->l_G->GCdebt > 0)
          luaC_step(L);
      }
      g->gcstp = oldstp;
      if (debt > 0 && g->gcstate == GCSpause)
        res = 1;
      break;
    }
    case LUA_GCSETPAUSE: {
      int data = va_arg(argp, int);
      res = getgcparam(g->gcpause);
      setgcparam(g->gcpause, data);
      break;
    }
    case LUA_GCSETSTEPMUL: {
      int data = va_arg(argp, int);
      res = getgcparam(g->gcstepmul);
      setgcparam(g->gcstepmul, data);
      break;
    }
    case LUA_GCISRUNNING: {
      res = gcrunning(g);
      break;
    }
    case LUA_GCGEN: {
      int minormul = va_arg(argp, int);
      int majormul = va_arg(argp, int);
      res = isdecGCmodegen(g) ? LUA_GCGEN : LUA_GCINC;
      if (minormul != 0)
        g->genminormul = minormul;
      if (majormul != 0)
        setgcparam(g->genmajormul, majormul);
      luaC_changemode(L, KGC_GEN);
      break;
    }
    case LUA_GCINC: {
      int pause = va_arg(argp, int);
      int stepmul = va_arg(argp, int);
      int stepsize = va_arg(argp, int);
      res = isdecGCmodegen(g) ? LUA_GCGEN : LUA_GCINC;
      if (pause != 0)
        setgcparam(g->gcpause, pause);
      if (stepmul != 0)
        setgcparam(g->gcstepmul, stepmul);
      if (stepsize != 0)
        g->gcstepsize = stepsize;
      luaC_changemode(L, KGC_INC);
      break;
    }
    default: res = -1;
  }
  va_end(argp);
  return res;
}
