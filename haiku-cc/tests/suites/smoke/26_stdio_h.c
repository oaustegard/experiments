#include <stdio.h>
#include <string.h>
#include <stdlib.h>
int main(void) { char buf[32]; snprintf(buf, sizeof buf, "%s-%d", "x", 9); char *p = malloc(8); strcpy(p, buf); puts(p); free(p); return 0; }
