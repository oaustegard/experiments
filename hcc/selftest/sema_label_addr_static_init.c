// EXPECT-EXIT: 3
int main(void) {
    static void *p[] = {&&a, &&b, &&c};
    int i = 0;
    goto *p[0];
a:  i++;
b:  i++;
c:  i++;
    return i;
}
