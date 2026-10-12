// EXPECT-EXIT: 0
int main(void) {
    int s = 0, i;
    for (i = 0; i < 10; i++) { if (i == 3) continue; if (i == 8) break; s += i; }
    if (s != 25) return 1;
    i = 0; while (i < 5) i++;
    if (i != 5) return 2;
    do { i--; } while (i > 0);
    if (i != 0) return 3;
    switch (s) { case 1: return 4; case 25: s = 100; break; default: return 5; }
    if (s != 100) return 6;
    switch (3) { case 1: case 3: s = 7; case 4: s++; break; }
    if (s != 8) return 7;
    i = 0;
again:
    i++;
    if (i < 3) goto again;
    return i == 3 ? 0 : 8;
}
