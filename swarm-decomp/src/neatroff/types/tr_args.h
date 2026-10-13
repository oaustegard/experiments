#pragma once
/* argument block of one troff request line: arg[0] is at offset 8 */
struct tr_args {
  long n;
  char *arg[8];
};
