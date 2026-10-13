#include <string.h>
#include "BuffFS.h"
#include "pushstr.h"

static void addstr2buff (BuffFS *buff, const char *str, size_t lstr) {
  int nb = buff->nb;
  if (lstr > BUFVFS - 1) {
    pushstr(buff, buff->buff, nb);
    buff->nb = 0;
    pushstr(buff, str, lstr);
  }
  else {
    if ((int)(nb + lstr) > BUFVFS - 1) {
      pushstr(buff, buff->buff, nb);
      buff->nb = 0;
      nb = 0;
    }
    memcpy(buff->buff + nb, str, lstr);
    buff->nb += lstr;
  }
}
