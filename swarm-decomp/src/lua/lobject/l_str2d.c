#include <string.h>
#include <locale.h>
#include "l_str2dloc.h"

#define L_MAXLENNUM 200

static const char *l_str2d (const char *s, lua_Number *result) {
  const char *endptr;
  const char *pmode = strpbrk(s, ".xXnN");
  int mode = pmode ? (((unsigned char)*pmode) | 0x20) : 0;
  if (mode == 'n')
    return NULL;
  endptr = l_str2dloc(s, result, mode);
  if (endptr == NULL) {
    char buff[L_MAXLENNUM + 1];
    const char *pdot = strchr(s, '.');
    if (pdot == NULL || strlen(s) > L_MAXLENNUM)
      return NULL;
    strcpy(buff, s);
    buff[pdot - s] = localeconv()->decimal_point[0];
    endptr = l_str2dloc(buff, result, mode);
    if (endptr != NULL)
      endptr = s + (endptr - buff);
  }
  return endptr;
}
