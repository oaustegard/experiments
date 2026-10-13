#include <string.h>
static const char *getobjname (const Proto *p, int lastpc, int reg, const char **name) {
  const char *kind = basicgetobjname(p, &lastpc, reg, name);
  if (kind != NULL)
    return kind;
  else if (lastpc != -1) {
    Instruction i = p->code[lastpc];
    int op = DGET_OPCODE(i);
    switch (op) {
      case DOP_GETTABUP: {
        int k = DGETARG_C(i);
        kname(p, k, name);
        return isEnv(p, lastpc, i, 1);
      }
      case DOP_GETTABLE: {
        int k = DGETARG_C(i);
        rname(p, lastpc, k, name);
        return isEnv(p, lastpc, i, 0);
      }
      case DOP_GETI: {
        *name = "integer index";
        return "field";
      }
      case DOP_GETFIELD: {
        int k = DGETARG_C(i);
        kname(p, k, name);
        return isEnv(p, lastpc, i, 0);
      }
      case DOP_SELF: {
        rkname(p, lastpc, i, name);
        return "method";
      }
      default: break;
    }
  }
  return NULL;
}
