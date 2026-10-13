#pragma once
#include "FuncState.h"
#include "expdesc.h"
#include "UnOpr.h"

void luaK_prefix (FuncState *fs, UnOpr op, expdesc *e, int line);
