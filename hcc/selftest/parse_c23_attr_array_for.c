// EXPECT-EXIT: 0
// C23 attributes after array dimensions and at the start of a for-init clause.
int g [2] [[]] [3] [[]];
int main(void) {
    int s = 0;
    for ([[]] int i = 0; i < 2; i++)
        s += i;
    return s == 1 && sizeof g == 6 * sizeof(int) ? 0 : 1;
}
