// EXPECT-EXIT: 0
// A stray non-UTF-8 byte in a macro argument is an ordinary token: #x keeps it.
#define str(x) #x
int main(void) {
    const char *s = str(a € b);
    const char *want = "a \200 b";
    int i = 0;
    while (s[i] && s[i] == want[i]) i++;
    return s[i] == want[i] ? 0 : 1;
}
