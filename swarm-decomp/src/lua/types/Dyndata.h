#pragma once
#include "Vardesc.h"
#include "Labellist.h"

typedef struct Dyndata {
  struct {
    Vardesc *arr;
    int n;
    int size;
  } actvar;
  Labellist gt;
  Labellist label;
} Dyndata;
