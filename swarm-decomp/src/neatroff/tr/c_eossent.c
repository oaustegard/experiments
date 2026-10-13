int c_eossent(char *s) { int i; for (i = 0; i < eos_sentcnt; i++) if (strcmp(eos_sent[i], s) == 0) return 1; return 0; }
