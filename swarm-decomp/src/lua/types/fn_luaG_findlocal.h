#pragma once
#include "lstate.h"

const char *luaG_findlocal (lua_State *L, CallInfo *ci, int n, StkId *pos);
