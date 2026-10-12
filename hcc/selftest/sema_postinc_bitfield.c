// EXPECT-EXIT: 0
// Post-increment/decrement yields the old value even when the store truncates
// (bitfields, _Bool): the old value must not be recomputed by reversing the update.
int main(void) {
    struct { unsigned int bit0 : 1; } s = {1};
    int c = (s.bit0-- > 0);       /* old value 1: true, field becomes 0 */
    int d = (s.bit0-- > 0);       /* old value 0: false, field wraps to 1 */
    _Bool b = 1;
    _Bool old = b++;              /* old value 1 */
    if (c != 1 || d != 0 || s.bit0 != 1) return 1;
    if (old != 1 || b != 1) return 2;
    return 0;
}
