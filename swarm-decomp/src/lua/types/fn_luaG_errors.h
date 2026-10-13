#pragma once
#include "lstate.h"
#include "l_noret.h"

l_noret luaG_ordererror (lua_State *L, const TValue *p1, const TValue *p2);
l_noret luaG_opinterror (lua_State *L, const TValue *p1, const TValue *p2, const char *msg);
l_noret luaG_tointerror (lua_State *L, const TValue *p1, const TValue *p2);
l_noret luaG_concaterror (lua_State *L, const TValue *p1, const TValue *p2);
