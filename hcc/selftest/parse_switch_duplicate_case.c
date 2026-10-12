// EXPECT-ERROR: [parse] duplicate case value
// Case values are converted to the promoted controlling type (int here),
// so -31 overlaps the range -32 ... -30.
int f(unsigned char x) {
    switch (x) {
    case -32 ... -30: return 1;
    case -31: return 2;
    }
    return 0;
}
