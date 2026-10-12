// EXPECT-EXIT: 7
// Two block-scope extern declarations of a file-scope object are allowed.
int g = 7;
int main(void) {
    extern int g;
    extern int g;
    return g;
}
