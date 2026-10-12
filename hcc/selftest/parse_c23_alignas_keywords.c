// EXPECT-EXIT: 16
// EXPECT-FLAGS: -std=c23
// C23 makes alignas, alignof and thread_local keywords (identifiers in other modes).
alignas (alignof (long double)) char c;
alignas (8) int i;
static thread_local int t;
int main(void) {
    return (int)alignof(int) + (int)alignof(c) * 0 + 12;
}
