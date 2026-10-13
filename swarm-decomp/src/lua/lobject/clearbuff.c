#include "clearbuff.h"

static void clearbuff (BuffFS *buff) {
  pushstr(buff, buff->buff, buff->nb);
  buff->nb = 0;
}
