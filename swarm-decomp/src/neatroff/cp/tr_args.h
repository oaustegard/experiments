#pragma once

char *tr_args(char **args, int endc, int (*next)(void), void (*back)(int));
