#pragma once
#include "lstate.h"
#include "cclosure.h"

CClosure *luaF_newCclosure (lua_State *L, int nupvals);
