#include <locale.h>
#include <stdio.h>
#include <string.h>
#include "tostringbuff.h"

static int tostringbuff (TValue *obj, char *buff) {
  int len;
  if (obj->tt_ == 3)
    len = snprintf(buff, 44, "%lld", (long long)obj->value_.i);
  else {
    len = snprintf(buff, 44, "%.14g", obj->value_.n);
    if (buff[strspn(buff, "-0123456789")] == '\0') {
      buff[len++] = *localeconv()->decimal_point;
      buff[len++] = '0';
    }
  }
  return len;
}
