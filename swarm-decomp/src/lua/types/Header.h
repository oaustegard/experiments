#pragma once
#include "lua_api.h"
typedef struct Header {
  lua_State *L;
  int islittle;
  int maxalign;
} Header;
