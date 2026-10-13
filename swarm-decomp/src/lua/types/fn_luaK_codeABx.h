#pragma once
#include "FuncState.h"
#include "OpCode.h"

int luaK_codeABx (FuncState *fs, OpCode o, int a, unsigned int bx);
