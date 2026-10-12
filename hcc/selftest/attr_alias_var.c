// EXPECT-EXIT: 0
static int target = 1;
extern int alias_name __attribute__((alias("target")));
int main(void) {
    alias_name = 5;
    return target != 5;
}
