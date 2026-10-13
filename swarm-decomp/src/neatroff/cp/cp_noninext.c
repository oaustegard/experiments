/* cp_noninext: next input character that is not a request-continuation marker */
static int cp_noninext(void)
{
  int c;

  do
    c = cp_next();
  while (c == 4);
  return c;
}
