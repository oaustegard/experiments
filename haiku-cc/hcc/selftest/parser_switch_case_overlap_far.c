// EXPECT-ERROR: duplicate
/* the overlapping earlier range (1...10) is not the last case written, so the
   overlap check must find it in the sorted case index, not just the last label. */
int main(void) {
    int x = 0;
    switch (x) {
    case 1 ... 10:
        break;
    case 20:
        break;
    case 5:
        break;
    }
    return 0;
}
