#pragma once
#include <stddef.h>
#include "regex_t.h"

int regcomp(regex_t *re, const char *pat, int cflags);
int regexec(const regex_t *re, const char *s, size_t nmatch, void *pmatch, int eflags);
