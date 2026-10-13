#pragma once
#include "lua_api.h"
#include "sys_signal.h"
#include "globals.h"

#define LUA_PROMPT "> "
#define LUA_PROMPT2 ">> "
#define LUA_MAXINPUT 512
#define EOFMARK "<eof>"
#define marklen (sizeof(EOFMARK) / sizeof(char) - 1)

static void l_message (const char *pname, const char *msg);
static void lstop (lua_State *L, lua_Debug *ar);
static int msghandler (lua_State *L);
static int pushline (lua_State *L, int firstline);
static void laction (int i);
static int docall (lua_State *L, int narg, int nres);
static void doREPL (lua_State *L);
static int dostring (lua_State *L, const char *s, const char *name);
static int pmain (lua_State *L);
static int report (lua_State *L, int status);
static int dofile (lua_State *L, const char *name);
static int handle_luainit (lua_State *L);
static int handle_script (lua_State *L, char **argv);
static int runargs (lua_State *L, char **argv, int n);
static int collectargs (char **argv, int *first);
static void createargtable (lua_State *L, char **argv, int argc, int script);
static void print_usage (const char *badoption);
static void print_version (void);
static int addreturn (lua_State *L);
static int incomplete (lua_State *L, int status);
static const char *get_prompt (lua_State *L, int firstline);
static int loadline (lua_State *L);
