#include "lauxlib.h"

static int io_close (lua_State *L);
static int io_flush (lua_State *L);
static int io_input (lua_State *L);
static int io_lines (lua_State *L);
static int io_open (lua_State *L);
static int io_output (lua_State *L);
static int io_popen (lua_State *L);
static int io_read (lua_State *L);
static int io_type (lua_State *L);
static int io_write (lua_State *L);
static int io_tmpfile (lua_State *L);

static const luaL_Reg iolib[] = {
  {"close", io_close},
  {"flush", io_flush},
  {"input", io_input},
  {"lines", io_lines},
  {"open", io_open},
  {"output", io_output},
  {"popen", io_popen},
  {"read", io_read},
  {"tmpfile", io_tmpfile},
  {"type", io_type},
  {"write", io_write},
  {NULL, NULL}
};
