#include <string.h>
#include "lstate.h"
#include "lua.h"
#include "Closure.h"
#include "fn_luaO_chunkid.h"
#define noLuaClosure(f)	((f) == NULL || (f)->c.tt != 6)
#define getstr(ts)	((ts)->contents)
#define tsslen(s)	((s)->shrlen != 0xff ? (s)->shrlen : (s)->u.lnglen)
#define LL(x)	(sizeof(x)/sizeof(char) - 1)

static void funcinfo (lua_Debug *ar, Closure *cl) {
  if (noLuaClosure(cl)) {
    ar->source = "=[C]";
    ar->srclen = LL("=[C]");
    ar->linedefined = -1;
    ar->lastlinedefined = -1;
    ar->what = "C";
  }
  else {
    const Proto *p = cl->l.p;
    if (p->source) {
      ar->source = getstr(p->source);
      ar->srclen = tsslen(p->source);
    }
    else {
      ar->source = "=?";
      ar->srclen = LL("=?");
    }
    ar->linedefined = p->linedefined;
    ar->lastlinedefined = p->lastlinedefined;
    ar->what = (ar->linedefined == 0) ? "main" : "Lua";
  }
  luaO_chunkid(ar->short_src, ar->source, ar->srclen);
}
