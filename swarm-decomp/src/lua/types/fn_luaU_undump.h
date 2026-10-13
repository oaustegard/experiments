#pragma once
#include "LClosure.h"
#include "Zio.h"

LClosure *luaU_undump (lua_State *L, ZIO *Z, const char *name);
