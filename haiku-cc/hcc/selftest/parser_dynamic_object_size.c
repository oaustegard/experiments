// EXPECT-EXIT: 0
// __builtin_dynamic_object_size is folded like __builtin_object_size for
// statically known objects instead of being emitted as an external call.
int g[5];
int main(void) {
    char *p = (char *)g;
    if (__builtin_dynamic_object_size(&g, 0) != sizeof g) return 1;
    if (__builtin_dynamic_object_size(g, 1) != sizeof g) return 2;
    if (__builtin_dynamic_object_size(p, 0) != (__SIZE_TYPE__)-1) return 3;
    if (__builtin_dynamic_object_size(p, 2) != 0) return 4;
    return 0;
}
