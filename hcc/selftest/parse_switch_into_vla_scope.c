// EXPECT-ERROR: [parse]
// A case label may not sit after a VLA declaration of the switch body that the switch skips.
int f(int a, int b) {
    switch (a) {
    int v[b];
    case 2:
        return 1;
    }
    return 0;
}
