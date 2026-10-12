// EXPECT-EXIT: 0
// Universal character names inside identifiers decode to their code points.
int a$b(void) { return 1; }
int Áx = 2;
int main(void) {
    if (a$b() != 1) return 1;
    if (\U000000C1x != 2) return 2;
    return 0;
}
