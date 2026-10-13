#pragma once
#include "lstate.h"

void luaT_getvarargs (lua_State *L, CallInfo *ci, StkId where, int wanted);
