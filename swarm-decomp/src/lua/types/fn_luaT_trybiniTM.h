#pragma once
#include "lstate.h"
#include "TMS.h"

void luaT_trybiniTM (lua_State *L, const TValue *p1, lua_Integer i2, int flip, StkId res, TMS event);
