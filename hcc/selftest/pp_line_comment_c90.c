// EXPECT-EXIT: 0
// `//` is a comment in gnu11 (default): this line must not be parsed as code.
int main(void) {
    int i = 4 //**/ 2
    ;
    return i == 4 ? 0 : 1;
}
