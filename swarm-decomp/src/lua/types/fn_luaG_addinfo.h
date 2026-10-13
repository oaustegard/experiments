#pragma once
#include "lobject.h"
const char *luaG_addinfo(lua_State *L, const char *msg, TString *src, int line);
