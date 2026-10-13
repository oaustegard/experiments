#pragma once
#include "lobject.h"
#include "LexState.h"
void luaX_setinput(lua_State *L, LexState *ls, ZIO *z, TString *source, int firstchar);
