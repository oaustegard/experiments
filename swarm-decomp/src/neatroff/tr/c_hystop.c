int c_hystop(char *s) { int i; for (i = 0; i < hy_stopcnt; i++) if (strcmp(hy_stop[i], s) == 0) return 1; return 0; }
