#pragma once
#include "Zio.h"
#include "Mbuffer.h"
#include "Dyndata.h"

struct SParser {
  ZIO *z;
  Mbuffer buff;
  Dyndata dyd;
  const char *mode;
  const char *name;
};
