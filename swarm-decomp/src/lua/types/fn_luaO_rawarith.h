#pragma once
#include "lstate.h"

int luaO_rawarith (lua_State *L, int op, const TValue *p1, const TValue *p2, StkId res);
