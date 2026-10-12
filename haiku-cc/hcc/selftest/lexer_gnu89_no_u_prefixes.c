// EXPECT-EXIT: 0
// EXPECT-FLAGS: -std=gnu89
// gnu89 has no u8/u/U literal prefixes: `u'a'` is identifier `u` plus a char constant.
#define u 1 +
int main(void) {
    int a = u'a';
    return a == 1 + 'a' ? 0 : 1;
}
