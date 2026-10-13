// EXPECT-EXIT: 7
// EXPECT-FLAGS: -std=c23
// C23: `auto` is not a storage class, so it combines with static/extern;
// `const auto` keeps the qualifier, while a plain `auto` drops the qualifiers of its initializer.
static auto u = 3U;
extern unsigned int u;
const auto k = 4LL;
extern const long long k;
int main(void) {
    const double cd = 1.0;
    auto x = k;
    x = 0;
    auto y = cd;
    y = 2.0;
    return (int) (u + k + x) + (int) (y - 2.0);
}
