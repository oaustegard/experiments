// EXPECT-EXIT: 0
// A label may be followed by a declaration whose type is a typedef name.
typedef unsigned int u32;
int main(void) {
    int i = 0;
    goto e;
e:
    u32 f = 5;
    i = (int)f;
    return i == 5 ? 0 : 1;
}
