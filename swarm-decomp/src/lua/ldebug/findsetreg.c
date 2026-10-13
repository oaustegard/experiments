static int findsetreg (const Proto *p, int lastpc, int reg) {
  int pc;
  int setreg = -1;
  int jmptarget = 0;
  if (DtestMMMode(DGET_OPCODE(p->code[lastpc])))
    lastpc--;
  for (pc = 0; pc < lastpc; pc++) {
    Instruction i = p->code[pc];
    int op = DGET_OPCODE(i);
    int a = DGETARG_A(i);
    int change;
    switch (op) {
      case DOP_JMP: {
        int b = DGETARG_sJ(i);
        int dest = pc + 1 + b;
        if (dest <= lastpc && dest > jmptarget)
          jmptarget = dest;
        change = 0;
        break;
      }
      case DOP_LOADNIL: {
        int b = DGETARG_B(i);
        change = (a <= reg && reg <= a + b);
        break;
      }
      case DOP_TFORCALL: {
        change = (reg >= a + 2);
        break;
      }
      case DOP_CALL:
      case DOP_TAILCALL: {
        change = (reg >= a);
        break;
      }
      default:
        change = (DtestAMode(op) && reg == a);
        break;
    }
    if (change)
      setreg = filterpc(pc, jmptarget);
  }
  return setreg;
}
