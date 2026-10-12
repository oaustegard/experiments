// EXPECT-EXIT: 7
// GNU: a struct member declaration may omit the ';' right before '}'
// (gcc warns "no semicolon at end of struct or union" and accepts it).
struct S0 {
    short f3;
    int f5
};
struct S0 g = { 3, 4 };
int main(void) {
    return g.f3 + g.f5;
}
