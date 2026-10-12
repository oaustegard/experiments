// EXPECT-EXIT: 0
static char a2x2[2][2];
extern char ext[];
char tent[];
int main(void) {
    if (__builtin_object_size(&a2x2[0][0], 1) != 2) return 1;
    if (__builtin_object_size(&a2x2[0][0], 0) != 4) return 2;
    if (__builtin_object_size(&a2x2[0] + 1, 0) != 2) return 3;
    if (__builtin_object_size(&a2x2[0] + 1, 2) != 2) return 4;
    if (__builtin_object_size(&a2x2[0][0] + 1, 3) != 0) return 5;
    if (__builtin_object_size(ext, 0) != (unsigned long)-1) return 6;
    if (__builtin_object_size(tent, 0) != 1) return 7;
    return 0;
}
