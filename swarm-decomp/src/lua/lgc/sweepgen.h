#pragma once
#include "lgc_defs.h"

static GCObject **sweepgen(lua_State *L, global_State *g, GCObject **p, GCObject *limit, GCObject **pfirstold1);
