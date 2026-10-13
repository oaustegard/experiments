#pragma once
#include "lstate.h"

_Noreturn void luaG_forerror (lua_State *L, const TValue *o, const char *what);
