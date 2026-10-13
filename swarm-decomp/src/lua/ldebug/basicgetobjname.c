static const char *basicgetobjname (const Proto *p, int *ppc, int reg, const char **name) {
  int pc = *ppc;
  *name = luaF_getlocalname(p, reg + 1, pc);
  if (*name)
    return "local";
  *ppc = pc = findsetreg(p, pc, reg);
  if (pc != -1) {
    Instruction i = p->code[pc];
    int op = DGET_OPCODE(i);
    switch (op) {
      case DOP_MOVE: {
        int b = DGETARG_B(i);
        if (b < DGETARG_A(i))
          return basicgetobjname(p, ppc, b, name);
        break;
      }
      case DOP_GETUPVAL: {
        *name = upvalname(p, DGETARG_B(i));
        return "upvalue";
      }
      case DOP_LOADK: return kname(p, DGETARG_Bx(i), name);
      case DOP_LOADKX: return kname(p, DGETARG_Ax(p->code[pc + 1]), name);
      default: break;
    }
  }
  return NULL;
}
