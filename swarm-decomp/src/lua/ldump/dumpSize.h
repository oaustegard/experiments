#pragma once
#include "DumpState.h"
#include "lu_byte.h"
#include "Proto.h"
static void dumpSize (DumpState *D, size_t x);
static void dumpBlock (DumpState *D, const void *b, size_t size);
static void dumpByte (DumpState *D, int y);
static void dumpInt (DumpState *D, int x);
static void dumpNumber (DumpState *D, lua_Number x);
static void dumpInteger (DumpState *D, lua_Integer x);
static void dumpCode (DumpState *D, const Proto *f);
static void dumpConstants (DumpState *D, const Proto *f);
static void dumpUpvalues (DumpState *D, const Proto *f);
static void dumpProtos (DumpState *D, const Proto *f);
static void dumpDebug (DumpState *D, const Proto *f);
static void dumpHeader (DumpState *D);
static void dumpFunction (DumpState *D, const Proto *f, TString *psource);
