#pragma once
#include "lstate.h"

int luaD_reallocstack (lua_State *L, int newsize, int raiseerror);
