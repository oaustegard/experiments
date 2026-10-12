// EXPECT-EXIT: 0
int main(void) {
    char c = *__builtin_strdup("x");    /* builtin signature: char *, not int */
    return c == 'x' ? 0 : 1;
}
