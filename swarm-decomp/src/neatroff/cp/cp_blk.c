/* cp_blk: skip blanks, or copy raw text up to the end of a block */
void cp_blk(int f)
{
  int c;

  if (f) {
    for (;;) {
      c = cp_raw();
      if (c < 0)
        return;
      if (c != '\n')
        continue;
      if (!(cp_blkdep > cp_reqdep))
        return;
    }
  }
  while ((c = cp_next()) == ' ')
    ;
  in_back(c);
}
