static int patchtestreg (FuncState *fs, int node, int reg) {
  Instruction *i = getjumpcontrol(fs, node);
  if ((*i & 0x7f) != OP_TESTSET)
    return 0;
  if (reg != NO_REG && reg != (int)((*i >> 16) & 0xff))
    *i = (*i & 0xffff807f) | (((Instruction)reg << 7) & 0x7f80);
  else
    *i = (OP_TEST) | (((*i >> 16) & 0xff) << 7) | ((*i >> 15) & 1) << 15;
  return 1;
}
