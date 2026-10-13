/* cp_raw: read one raw character of copy mode (c_ec escapes, braces, newline) */
static int cp_raw(void)
{
  int c;

  for (;;) {
    if (in_top() >= 0)
      return in_next();
    do
      c = in_next();
    while (c == 4);
    if (c != c_ec)
      return c;
    do
      c = in_next();
    while (c == 4);
    if (c > 'a') {
      if (c == 't') {
        in_back(9);
        return 4;
      }
      if (c == '}') {
        if (!cp_cpmode) {
          cp_blkdep = cp_blkdep - 1;
          return ' ';
        }
      } else if (c == '{') {
        if (!cp_cpmode) {
          cp_blkdep = cp_blkdep + 1;
          return ' ';
        }
      }
    } else if (c == 'a') {
      in_back(1);
      return 4;
    } else {
      if (c == '.')
        return c;
      if (c == '\\') {
        in_back('\\');
        return 4;
      }
      if (c == '\n')
        continue;
    }
    in_back(c);
    return c_ec;
  }
}
