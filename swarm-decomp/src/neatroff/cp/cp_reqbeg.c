/* cp_reqbeg: remember the block depth at the start of a request */
void cp_reqbeg(void)
{
  cp_reqdep = cp_blkdep;
}
