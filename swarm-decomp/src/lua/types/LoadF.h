#pragma once
#include <stdio.h>

/* lauxlib's file-loading state: pending count, file, and read buffer */
typedef struct LoadF {
  int n;  /* number of pre-read characters */
  FILE *f;  /* file being read */
  char buff[BUFSIZ];  /* area for reading file */
} LoadF;
