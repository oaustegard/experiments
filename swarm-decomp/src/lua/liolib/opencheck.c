#include "opencheck.h"
#include "newfile.h"
#include "io_fclose.h"
#include <errno.h>
#include <string.h>

static void opencheck (lua_State *L, const char *fname, const char *mode) {
  luaL_Stream *p = newfile(L);
  p->closef = &io_fclose;
  p->f = fopen(fname, mode);
  if (__builtin_expect(p->f == NULL, 0))
    luaL_error(L, "cannot open file '%s' (%s)", fname, strerror(errno));
}
