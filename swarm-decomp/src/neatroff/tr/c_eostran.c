int c_eostran(char *s) { int i; for (i = 0; i < eos_trancnt; i++) if (strcmp(eos_tran[i], s) == 0) return 1; return 0; }
