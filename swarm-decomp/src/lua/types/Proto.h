#pragma once
#include "lu_byte.h"
#include "ls_byte.h"
#include "Instruction.h"
#include "AbsLineInfo.h"
#include "lobject.h"

struct TValue;
struct Proto;
struct Upvaldesc;
struct LocVar;
struct TString;
struct GCObject;

typedef struct Proto {
  struct GCObject *next;
  lu_byte tt;
  lu_byte marked;
  lu_byte numparams;
  lu_byte is_vararg;
  lu_byte maxstacksize;
  int sizeupvalues;
  int sizek;
  int sizecode;
  int sizelineinfo;
  int sizep;
  int sizelocvars;
  int sizeabslineinfo;
  int linedefined;
  int lastlinedefined;
  struct TValue *k;
  Instruction *code;
  struct Proto **p;
  struct Upvaldesc *upvalues;
  ls_byte *lineinfo;
  AbsLineInfo *abslineinfo;
  struct LocVar *locvars;
  struct TString *source;
  struct GCObject *gclist;
} Proto;
