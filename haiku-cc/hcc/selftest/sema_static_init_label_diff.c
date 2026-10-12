// EXPECT-EXIT: 0
// Static initializers: label differences (&&a - &&b) as relocs, and a
// file-scope compound literal used as a struct initializer.
struct empty_s { };
struct s { unsigned char a; struct empty_s e; unsigned char b; };
struct s gs = { 1, (struct empty_s){}, 3 };

int f(int x) {
    static int tab[] = { &&lab1 - &&lab0, &&lab2 - &&lab0 };
    if (x == 0) goto lab0;
    if (x == 1) goto lab1;
    if (x == 3) goto lab2;
    return tab[1] == 0 ? 2 : 3;
lab0:
    return 0;
lab1:
    return tab[0] == 0 ? 4 : 5;
lab2:
    return 1;
}

int main(void) {
    int bad = 0;
    bad += gs.a != 1 || gs.b != 3;
    bad += f(0) != 0;
    bad += f(1) != 5;
    bad += f(2) != 3;
    bad += f(3) != 1;
    return bad;
}
