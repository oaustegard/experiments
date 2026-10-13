#include <string.h>
#include "BuffFS.h"
#include "pushstr.h"

static void addstr2buff (BuffFS *buff, const char *str, size_t lstr) {
  int nb = buff->nb;
  int len = (int)lstr;
  int avail = BUFVFS - 1 - nb;
  if (lstr <= BUFVFS - 1) {
    if (len > avail) {
      pushstr(buff, buff->buff, nb);
      buff->nb = 0;
      nb = 0;
    }
    memcpy(buff->buff + nb, str, lstr);
    buff->nb += len;
    return;
  }
  pushstr(buff, buff->buff, nb);
  buff->nb = 0;
  pushstr(buff, str, lstr);
}
