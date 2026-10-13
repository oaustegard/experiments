#include "outelf.h"

static long hexnum(char *s)
{
    long v = 0;
    int d;
    if (*s == '0' && s[1] == 'x')
        s += 2;
    while (isalpha(*s) || isdigit(*s)) {
        v <<= 4;
        if (isdigit(*s))
            d = *s - '0';
        else
            d = tolower(*s) - 'a' + 10;
        v |= d;
        s++;
    }
    return v;
}
