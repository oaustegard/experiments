// EXPECT-EXIT: 0
// tls_model with a valid model name is accepted.
__thread int c __attribute__((tls_model("initial-exec")));
int main(void) { c = 3; return c - 3; }
