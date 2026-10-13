#pragma once
#include "lobject.h"
#include "Proto.h"

struct Proto;
struct UpVal;

typedef struct LClosure {
  GCObject *next;
  lu_byte tt;
  lu_byte marked;
  lu_byte nupvalues;
  GCObject *gclist;
  struct Proto *p;
  struct UpVal *upvals[1];
} LClosure;
