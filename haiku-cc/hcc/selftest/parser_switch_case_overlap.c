// EXPECT-ERROR: duplicate
/* case labels are converted to the promoted switch type first, so -31 and
   the wrapped value of -32...-30 on an unsigned char switch collide. */
int main(void) {
    unsigned char x = 0;
    switch (x) {
    case -32 ... -30:
        break;
    case -31:
        break;
    }
    return 0;
}
