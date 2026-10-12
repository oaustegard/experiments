// EXPECT-EXIT: 0
// Library integer abs functions are expanded inline even when the program
// declares its own (wrong) definition, as GCC does.
long long llabs(long long);
void abort(void);
long long llabs(long long b) { abort(); return b; }
int main(void) {
    long long a = -5;
    int i = -7;
    if (llabs(a) != 5) return 1;
    if (llabs(a) + 0 != 5) return 2;
    return 0;
}
