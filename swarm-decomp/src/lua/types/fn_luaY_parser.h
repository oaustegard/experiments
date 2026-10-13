#pragma once
#include "LClosure.h"
#include "Zio.h"
#include "Mbuffer.h"
#include "Dyndata.h"

LClosure *luaY_parser (lua_State *L, ZIO *z, Mbuffer *buff, Dyndata *dyd, const char *name, int firstchar);
