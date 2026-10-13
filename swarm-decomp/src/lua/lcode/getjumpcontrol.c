static Instruction *getjumpcontrol (FuncState *fs, int pc) {
  Instruction *pi = &fs->f->code[pc];
  if (pc >= 1 && (luaP_opmodes[((*(pi - 1)) & 0x7f)] & (1 << 4)))
    return pi - 1;
  else
    return pi;
}
