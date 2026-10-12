// EXPECT-EXIT: 0
// Static initializers fold x/0.0 with IEEE semantics: 0/0 is NaN, 1/0 is +inf.
static double nan_tab[] = { 0.0 / 0.0 };
static double inf_tab[] = { 1.0 / 0.0, -1.0 / 0.0 };
int main(void) {
    if (nan_tab[0] == nan_tab[0]) return 1;
    if (inf_tab[0] != 1.0 / 0.0 || inf_tab[1] != -1.0 / 0.0) return 2;
    return 0;
}
