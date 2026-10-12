// EXPECT-EXIT: 0
/* switch on a 128-bit value compares both words of the case constant. */
typedef unsigned __int128 u128;
int main(void) {
    u128 v = ((u128)3 << 64) + 9;
    switch (v) {
    case 9: return 1;
    case ((u128)3 << 64) + 9: break;
    default: return 2;
    }
    switch ((u128)9) {
    case 9: return 0;
    default: return 3;
    }
}
