#pragma once
/* growable string buffer, 16 bytes */
struct sbuf {
  char *buf;
  int cap;
  int len;
};
