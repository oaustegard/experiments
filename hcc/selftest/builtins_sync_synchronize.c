// EXPECT-EXIT: 0
// __sync_synchronize() is a zero-argument full memory barrier (no libgcc call).
static int g = 3;
int main(void) {
    __sync_synchronize();
    g = 4;
    __sync_synchronize();
    return g == 4 ? 0 : 1;
}
