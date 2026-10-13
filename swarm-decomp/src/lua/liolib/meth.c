#include "lauxlib.h"

static int f_read (lua_State *L);
static int f_write (lua_State *L);
static int f_flush (lua_State *L);
static int f_seek (lua_State *L);
static int f_close (lua_State *L);
static int f_setvbuf (lua_State *L);
static int f_lines (lua_State *L);

static const luaL_Reg meth[] = {
  {"read", f_read},
  {"write", f_write},
  {"lines", f_lines},
  {"flush", f_flush},
  {"seek", f_seek},
  {"close", f_close},
  {"setvbuf", f_setvbuf},
  {NULL, NULL}
};
