// EXPECT-EXIT: 0
// Tentative (uninitialized, non-static) globals are common symbols; a second
// tentative definition of the same name must be legal in one translation unit.
int g;
int g = 4;
int h;
int main(void) {
    h = 2;
    return g + h - 6;
}
