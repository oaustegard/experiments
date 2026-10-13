#pragma once
#include "lstate.h"

void luaD_seterrorobj (lua_State *L, int errcode, StkId oldtop);
