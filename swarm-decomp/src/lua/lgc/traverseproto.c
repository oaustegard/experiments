#include "traverseproto.h"
#include "reallymarkobject.h"

static int traverseproto(global_State *g, Proto *f) {
  int i;
  markobjectN(g, f->source);
  for (i = 0; i < f->sizek; i++)
    markvalue(g, &f->k[i]);
  for (i = 0; i < f->sizeupvalues; i++)
    markobjectN(g, f->upvalues[i].name);
  for (i = 0; i < f->sizep; i++)
    markobjectN(g, f->p[i]);
  for (i = 0; i < f->sizelocvars; i++)
    markobjectN(g, f->locvars[i].varname);
  return 1 + f->sizek + f->sizeupvalues + f->sizep + f->sizelocvars;
}
