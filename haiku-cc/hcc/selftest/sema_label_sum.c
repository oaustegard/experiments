// EXPECT-EXIT: 0
// Static initializers that are sums/differences of label differences
// ((&&a-&&b)+(&&c-&&d), (&&a-&&b)-k) are emitted as assembler-resolved relocs.
int f(void) {
    static long a = ((char *)&&l1 - (char *)&&l2) + ((char *)&&l3 - (char *)&&l2) - 2;
    static long b = ((char *)&&l3 - (char *)&&l1) - ((char *)&&l2 - (char *)&&l1) + 7;
    static int c = (char *)&&l2 - (char *)&&l1;
    int bad = 0;
l1:
    bad |= 0;
l2:
    bad |= 0;
l3:
    if (c != (char *)&&l2 - (char *)&&l1) bad = 1;
    if (b != (char *)&&l3 - (char *)&&l2 + 7) bad = 1;
    if (a != ((char *)&&l1 - (char *)&&l2) + ((char *)&&l3 - (char *)&&l2) - 2) bad = 1;
    return bad;
}

int main(void) { return f(); }
