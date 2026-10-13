#include <stdio.h>

static int skipcomment (FILE *f, int *cp) {
  int c = getc(f);
  if (c == 0xEF) {
    if (getc(f) == 0xBB && getc(f) == 0xBF)
      c = getc(f);
    else {
      *cp = 0xEF;
      c = 0xEF;
    }
  }
  *cp = c;
  if (c != '#')
    return 0;
  do {
    c = getc(f);
  } while (c != EOF && c != '\n');
  *cp = getc(f);
  return 1;
}
