#include <string.h>

#include "lauxlib.h"
#include "getnextfilename.h"

static const char *getnextfilename (char **path, char *end) {
  char *sep;
  char *name = *path;
  if (name == end)
    return NULL;  /* no more names */
  else if (*name == '\0') {  /* from previous iteration? */
    *name = *LUA_PATH_SEP;  /* restore separator */
    name++;  /* skip it */
  }
  sep = strchr(name, *LUA_PATH_SEP);  /* find next separator */
  if (sep == NULL)  /* separator not found? */
    sep = end;  /* name goes until the end of the string */
  *sep = '\0';  /* finish file name */
  *path = sep;  /* will start next search from here */
  return name;
}
