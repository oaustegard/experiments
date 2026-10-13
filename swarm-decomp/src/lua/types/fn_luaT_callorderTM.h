#pragma once
#include "lstate.h"
#include "TMS.h"

int luaT_callorderTM (lua_State *L, const TValue *p1, const TValue *p2, TMS event);
