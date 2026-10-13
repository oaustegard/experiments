#pragma once
#include <stdio.h>

#define L_MAXLENNUM 200

typedef struct RN {
  FILE *f;
  int c;
  int n;
  char buff[L_MAXLENNUM + 1];
} RN;
