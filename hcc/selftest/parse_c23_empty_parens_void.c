// EXPECT-EXIT: 0
// EXPECT-FLAGS: -std=c23
// C23: `f()` declares a prototype with (void); _Generic must see it as void(void).
void f3();
int main(void) {
    return _Generic(f3, void (*)(int): 1, void (*)(void): 0, default: 3);
}
