// EXPECT-EXIT: 0
// Static initializer: a sum of two label differences whose shared label cancels,
// (&&l2-&&l3)+(&&l1-&&l2), is an assembler-resolved reloc for &&l1-&&l3.
int f(void) {
    static unsigned long a = ((char *)&&l2 - (char *)&&l3) + ((char *)&&l1 - (char *)&&l2);
    unsigned long want = (char *)&&l1 - (char *)&&l3;
l1:
l2:
l3:
    return a == want;
}
int main(void) { return f() ? 0 : 1; }
