#include <string.h>
#include "TString.h"
#include "lobject_tags.h"

static int l_strcmp (const TString *ls, const TString *rs) {
  const char *l = ls->contents;
  size_t ll = (ls->shrlen != 0xff) ? ls->shrlen : ls->u.lnglen;
  const char *r = rs->contents;
  size_t lr = (rs->shrlen != 0xff) ? rs->shrlen : rs->u.lnglen;
  for (;;) {
    int temp = strcoll(l, r);
    if (temp != 0)
      return temp;
    else {
      size_t len = strlen(l);
      size_t lenr = strlen(r);
      if (lr == lenr)
        return (len == ll) ? 0 : 1;
      else if (len == ll)
        return -1;
      len++;
      lenr++;
      l += len; ll -= len; r += lenr; lr -= lenr;
    }
  }
}
