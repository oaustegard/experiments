#include "dumpSize.h"
#include "Proto.h"
#include "fn_luaU_dump.h"
int luaU_dump (lua_State *L, const Proto *f, lua_Writer w, void *data, int strip) {
  DumpState D;
  D.L = L;
  D.writer = w;
  D.data = data;
  D.strip = strip;
  D.status = 0;
  dumpHeader(&D);
  dumpByte(&D, f->sizeupvalues);
  dumpFunction(&D, f, NULL);
  return D.status;
}
