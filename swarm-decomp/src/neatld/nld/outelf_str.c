#include "outelf.h"

static int outelf_str(struct outelf *oe, char *s)
{
    int n = strlen(s) + 1;
    int off = oe->symstr_len;
    while (oe->symstr_len + n > oe->symstr_alloc) {
        int a = oe->symstr_alloc ? oe->symstr_alloc * 2 : 0x400;
        oe->symstr_alloc = a;
        oe->symstr = realloc(oe->symstr, a);
        if (!oe->symstr)
            die("neatld: failed to allocate oe->symstr");
    }
    memcpy(oe->symstr + oe->symstr_len, s, n);
    oe->symstr_len += n;
    return off;
}
