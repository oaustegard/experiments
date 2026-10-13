#pragma once
#include "MatchState.h"

typedef struct GMatchState {
  const char *src;
  const char *p;
  const char *lastmatch;
  MatchState ms;
} GMatchState;
