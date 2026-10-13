#include "getbuff.h"
#include "clearbuff.h"

static char *getbuff (BuffFS *buff, int sz) {
  if (sz > BUFVFS - 1 - buff->nb)
    clearbuff(buff);
  return buff->buff + buff->nb;
}
