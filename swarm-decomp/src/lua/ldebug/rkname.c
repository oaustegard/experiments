static void rkname (const Proto *p, int pc, Instruction i, const char **name) {
  int c = DGETARG_C(i);
  if (DGETARG_k(i))
    kname(p, c, name);
  else
    rname(p, pc, c, name);
}
