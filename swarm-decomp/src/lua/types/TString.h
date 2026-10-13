#pragma once
#include <stddef.h>
#include "lobject.h"
#include "lu_byte.h"
#include "GCObject.h"

typedef struct TString TString;

struct TString {
  GCObject *next;
  lu_byte tt;
  lu_byte marked;
  lu_byte extra;
  lu_byte shrlen;
  unsigned int hash;
  union {
    size_t lnglen;
    struct TString *hnext;
  } u;
  char contents[1];
};
