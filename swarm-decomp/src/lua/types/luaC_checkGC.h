#pragma once
#include "G.h"
#include "fn_luaC_step.h"

#define luaC_checkGC(L) { if (G(L)->GCdebt > 0) luaC_step(L); }
