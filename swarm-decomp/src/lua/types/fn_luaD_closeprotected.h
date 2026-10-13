#pragma once
#include "lstate.h"

int luaD_closeprotected(lua_State *L, ptrdiff_t level, int status);
