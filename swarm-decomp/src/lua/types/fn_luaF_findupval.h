#pragma once
#include "lstate.h"
#include "UpVal.h"

UpVal *luaF_findupval (lua_State *L, StkId level);
