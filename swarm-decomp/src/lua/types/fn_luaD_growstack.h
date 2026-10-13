#pragma once
#include "lstate.h"

int luaD_growstack (lua_State *L, int n, int raiseerror);
