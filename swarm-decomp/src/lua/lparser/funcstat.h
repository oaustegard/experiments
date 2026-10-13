#pragma once
#pragma once
#include "LexState.h"
#include "FuncState.h"
#include "BlockCnt.h"
#include "Dyndata.h"
#include "expdesc.h"
#include "Labellist.h"
#include "Labeldesc.h"
#include "TString.h"
static int funcname (LexState *ls, expdesc *v);
static void funcstat (LexState *ls, int line);
