#pragma once
#include "lstate.h"

_Noreturn void luaG_typeerror (lua_State *L, const TValue *o, const char *op);
