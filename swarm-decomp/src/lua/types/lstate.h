#pragma once
#include "lobject.h"
#include "TString.h"
#include "TMS.h"
#include "Table.h"

#define STRCACHE_N 53
#define STRCACHE_M 2

typedef struct CallInfo {
  StkId func;
  StkId top;
  struct CallInfo *previous, *next;
  union {
    struct {
      const unsigned int *savedpc;
      volatile int trap;
      int nextraargs;
    } l;
    struct {
      lua_KFunction k;
      ptrdiff_t old_errfunc;
      lua_KContext ctx;
    } c;
  } u;
  union {
    int funcidx;
    int nyield;
    int nres;
    struct {
      unsigned short ftransfer;
      unsigned short ntransfer;
    } transferinfo;
  } u2;
  short nresults;
  unsigned short callstatus;
} CallInfo;

typedef struct stringtable {
  struct TString **hash;
  int nuse;
  int size;
} stringtable;

typedef struct global_State {
  lua_Alloc frealloc;
  void *ud;
  l_mem totalbytes;
  l_mem GCdebt;
  lu_mem GCestimate;
  lu_mem lastatomic;
  stringtable strt;
  TValue l_registry;
  TValue nilvalue;
  unsigned int seed;
  lu_byte currentwhite;
  lu_byte gcstate;
  lu_byte gckind;
  lu_byte gcstopem;
  lu_byte genminormul;
  lu_byte genmajormul;
  lu_byte gcstp;
  lu_byte gcemergency;
  lu_byte gcpause;
  lu_byte gcstepmul;
  lu_byte gcstepsize;
  GCObject *allgc;
  GCObject **sweepgc;
  GCObject *finobj;
  GCObject *gray;
  GCObject *grayagain;
  GCObject *weak;
  GCObject *ephemeron;
  GCObject *allweak;
  GCObject *tobefnz;
  GCObject *fixedgc;
  GCObject *survival;
  GCObject *old1;
  GCObject *reallyold;
  GCObject *firstold1;
  GCObject *finobjsur;
  GCObject *finobjold1;
  GCObject *finobjrold;
  struct lua_State *twups;
  lua_CFunction panic;
  struct lua_State *mainthread;
  TString *memerrmsg;
  TString *tmname[TM_N];
  struct Table *mt[LUA_NUMTYPES];
  TString *strcache[STRCACHE_N][STRCACHE_M];
  lua_WarnFunction warnf;
  void *ud_warn;
} global_State;

struct lua_State {
  GCObject *next;
  lu_byte tt;
  lu_byte marked;
  lu_byte status;
  lu_byte allowhook;
  unsigned short nci;
  StkId top;
  global_State *l_G;
  CallInfo *ci;
  StkId stack_last;
  StkId stack;
  struct UpVal *openupval;
  StkId tbclist;
  GCObject *gclist;
  struct lua_State *twups;
  void *errorJmp;
  CallInfo base_ci;
  volatile lua_Hook hook;
  ptrdiff_t errfunc;
  unsigned int nCcalls;
  int oldpc;
  int basehookcount;
  int hookcount;
  volatile int hookmask;
};
