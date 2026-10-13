#pragma once
#include "lstate.h"
#include "ltags.h"

const TValue *luaT_gettmbyobj (lua_State *L, const TValue *o, TMS event);
