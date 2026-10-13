#pragma once
#include "lua_api.h"

typedef struct DumpState {
  lua_State *L;
  lua_Writer writer;
  void *data;
  int strip;
  int status;
} DumpState;
