#pragma once
#include "Pfunc.h"

int luaD_rawrunprotected (lua_State *L, Pfunc f, void *ud);
