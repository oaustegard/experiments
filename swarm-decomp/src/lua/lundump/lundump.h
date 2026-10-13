#pragma once
#include <limits.h>
#include <stddef.h>
#include <string.h>
#include "LoadState.h"
#include "Proto.h"
#include "TString.h"
#include "LClosure.h"
#include "Upvaldesc.h"
#include "LocVar.h"
#include "AbsLineInfo.h"

/* static helpers of lundump.c; the bodies live in their own files */
static void error (LoadState *S, const char *why);
static lu_byte loadByte (LoadState *S);
static size_t loadUnsigned (LoadState *S, size_t limit);
static size_t loadSize (LoadState *S);
static int loadInt (LoadState *S);
static void loadBlock (LoadState *S, void *b, size_t size);
static void fchecksize (LoadState *S, size_t size, const char *tname);
static void checkliteral (LoadState *S, const char *s, const char *msg);
static void checkHeader (LoadState *S);
static lua_Number loadNumber (LoadState *S);
static lua_Integer loadInteger (LoadState *S);
static TString *loadStringN (LoadState *S, Proto *p);
static TString *loadString (LoadState *S, Proto *p);
static void loadFunction (LoadState *S, Proto *f, TString *psource);
static void loadCode (LoadState *S, Proto *f);
static void loadConstants (LoadState *S, Proto *f);
static void loadUpvalues (LoadState *S, Proto *f);
static void loadProtos (LoadState *S, Proto *f);
static void loadDebug (LoadState *S, Proto *f);

#define loadVector(S,b,n)	loadBlock(S,b,(n)*sizeof((b)[0]))
