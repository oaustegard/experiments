// EXPECT-EXIT: 0
struct empty {};
struct empty get(void) {
    struct empty e;
    return e;
}
int main(void) {
    get();
    return 0;
}
