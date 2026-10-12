// EXPECT-EXIT: 0
int g[5] = {1, 2, 3, 4, 5};
int *gp = &g[2];
char msg[] = "hey";
int main(void) {
    int a[3][4], i, j, *p;
    for (i = 0; i < 3; i++) for (j = 0; j < 4; j++) a[i][j] = i * 4 + j;
    if (a[2][3] != 11) return 1;
    p = &a[1][0];
    if (*(p + 2) != 6) return 2;
    if (&a[2][0] - &a[0][0] != 8) return 3;
    if (*gp != 3 || gp[1] != 4) return 4;
    if (sizeof(a) != 48 || sizeof msg != 4) return 5;
    if (msg[1] != 'e') return 6;
    char *s = "abc";
    if (s[2] != 'c' || *s != 97) return 7;
    return 0;
}
