#pragma once
#include "lstate.h"

void luaD_callnoyield(lua_State *L, StkId func, int nResults);
