#pragma once
#include "lstate.h"

CallInfo *luaD_precall (lua_State *L, StkId func, int nret);
