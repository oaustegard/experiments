#pragma once
#include "lstate.h"
#include "TMS.h"

int luaT_callorderiTM (lua_State *L, const TValue *p1, int v2, int flip, int isfloat, TMS event);
