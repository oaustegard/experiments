#pragma once
#include "Table.h"
#include "TString.h"
#include "lstate.h"
#include "TMS.h"

const TValue *luaT_gettm (Table *events, TMS event, TString *ename);
