#pragma once
#include "FuncState.h"
#include "expdesc.h"
#include "TValue.h"

int luaK_exp2const (FuncState *fs, const expdesc *e, TValue *v);
