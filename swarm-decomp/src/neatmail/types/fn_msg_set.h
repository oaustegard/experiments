#pragma once
#include <stddef.h>

int msg_set(char *s, size_t n, char **out, long *outlen, char *key, char *val);
