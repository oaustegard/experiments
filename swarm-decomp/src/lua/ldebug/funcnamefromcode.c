static const char *funcnamefromcode (lua_State *L, const Proto *p, int pc, const char **name) {
  TMS tm = (TMS)0;
  Instruction i = p->code[pc];
  switch (DGET_OPCODE(i)) {
    case DOP_CALL:
    case DOP_TAILCALL:
      return getobjname(p, pc, DGETARG_A(i), name);
    case DOP_TFORCALL: {
      *name = "for iterator";
      return "for iterator";
    }
    case DOP_SELF: case DOP_GETTABUP: case DOP_GETTABLE:
    case DOP_GETI: case DOP_GETFIELD:
      tm = TM_INDEX;
      break;
    case DOP_SETTABUP: case DOP_SETTABLE: case DOP_SETI: case DOP_SETFIELD:
      tm = TM_NEWINDEX;
      break;
    case DOP_MMBIN: case DOP_MMBINI: case DOP_MMBINK:
      tm = (TMS)DGETARG_C(i);
      break;
    case DOP_UNM: tm = TM_UNM; break;
    case DOP_BNOT: tm = TM_BNOT; break;
    case DOP_LEN: tm = TM_LEN; break;
    case DOP_CONCAT: tm = TM_CONCAT; break;
    case DOP_EQ: tm = TM_EQ; break;
    case DOP_LT: case DOP_LTI: case DOP_GTI: tm = TM_LT; break;
    case DOP_LE: case DOP_LEI: case DOP_GEI: tm = TM_LE; break;
    case DOP_CLOSE: case DOP_RETURN: tm = TM_CLOSE; break;
    default: return NULL;
  }
  *name = L->l_G->tmname[tm]->contents + 2;
  return "metamethod";
}
