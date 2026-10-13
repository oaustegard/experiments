static int getjump (FuncState *fs, int pc) {
  int offset = (int)(fs->f->code[pc] >> 7) - OFFSET_sJ;
  if (offset == NO_JUMP)
    return NO_JUMP;
  else
    return (pc + 1) + offset;
}
