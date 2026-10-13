#pragma once
#include "lstate.h"

void luaT_adjustvarargs (lua_State *L, int nfixparams, CallInfo *ci, const struct Proto *p);
