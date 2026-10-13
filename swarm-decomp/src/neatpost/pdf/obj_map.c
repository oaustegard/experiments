#include "fn_mextend.h"
static int obj_map(void) {
    int r;
    if (obj_n == obj_sz) {
        obj_sz = obj_n + 0x400;
        obj_off = mextend(obj_off, obj_n, obj_sz, 4);
    }
    r = obj_n;
    obj_n = r + 1;
    return r;
}
