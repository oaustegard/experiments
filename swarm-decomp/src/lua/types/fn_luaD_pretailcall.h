#pragma once
#include "lstate.h"

int luaD_pretailcall (lua_State *L, CallInfo *ci, StkId func, int narg1, int delta);
