#pragma once
#include "lstate.h"
#include "Proto.h"

int luaU_dump (lua_State *L, const Proto *f, lua_Writer w, void *data, int strip);
