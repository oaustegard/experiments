#pragma once
#include "lstate.h"

void luaC_runtilstate(lua_State *L, int statesmask);
