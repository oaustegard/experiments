// EXPECT-EXIT: 0
// A packed enum takes the smallest integer size holding its values; the
// declaration of a signed char function is compatible with it.
enum F { F1 = -1, F2, F3 } __attribute__ ((__packed__));
enum F fn1(void);
signed char fn1(void) { return F2; }
int main(void) {
    return sizeof(enum F) == 1 && fn1() == 0 ? 0 : 1;
}
