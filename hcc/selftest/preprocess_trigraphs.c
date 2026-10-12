// EXPECT-EXIT: 0
// Default (gnu) mode: trigraphs are NOT replaced, so "??=" stays three chars.
// Strict -std=c11 / -trigraphs replacement is covered by tests/gcc-dg/cpp.
int main(void) {
    const char *s = "??=";
    return s[0] == '?' && s[1] == '?' && s[2] == '=' ? 0 : 1;
}
