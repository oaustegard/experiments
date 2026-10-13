#pragma once
#include "lstate.h"

void luaO_arith (lua_State *L, int op, const TValue *p1, const TValue *p2, StkId res);
