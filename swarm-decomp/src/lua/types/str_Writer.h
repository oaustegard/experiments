#pragma once
#include "lua_api.h"
struct str_Writer {
  int init;
  luaL_Buffer B;
};
