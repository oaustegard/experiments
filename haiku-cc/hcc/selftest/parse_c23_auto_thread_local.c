// EXPECT-EXIT: 8
// EXPECT-FLAGS: -std=c23
// C23: `auto` may wrap its name in parentheses and take [[attrs]];
// `thread_local` is a keyword (= _Thread_local) in C23 mode.
auto ((gi)) = 3;
thread_local auto gf = 1.0f;
int main(void) {
    auto ((la)) = 2;
    auto ai [[]] = 1;
    static thread_local int t = 1;
    return gi + la + ai + t + (int) gf;   /* 3 + 2 + 1 + 1 + 1 */
}
