// EXPECT-ERROR: [parse] switch jumps into statement expression
// A case label inside a statement expression whose switch is outside it is invalid.
void f(int a) {
    switch (a) {
    case 0:
        ({
        case 1:
            ;
        0;
        });
    }
}
