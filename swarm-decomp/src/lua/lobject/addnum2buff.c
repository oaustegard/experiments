#include <stdio.h>
#include "BuffFS.h"
#include "pushstr.h"
#include "tostringbuff.h"



static void addnum2buff (BuffFS *buff, TValue *num) {
  int nb = buff->nb;
  char *numbuff;
  int len;
  if (44 > BUFVFS - 1 - nb) {
    pushstr(buff, buff->buff, nb);
    buff->nb = 0;
    nb = 0;
  }
  numbuff = buff->buff + nb;
  if (num->tt_ == 3)
    len = snprintf(numbuff, 44, "%lld", (long long)num->value_.i);
  else
    len = tostringbuff(num, numbuff);
  buff->nb += len;
}
