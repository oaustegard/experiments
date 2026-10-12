// EXPECT-EXIT: 5
// Outside C23 mode `constexpr` is an ordinary identifier (keyword only with -std=c23).
int main(void) {
    int constexpr = 5;
    return constexpr;
}
