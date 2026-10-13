int c_hydash(char *s) { int i; for (i = 0; i < hy_dashcnt; i++) if (strcmp(hy_dash[i], s) == 0) return 1; return 0; }
