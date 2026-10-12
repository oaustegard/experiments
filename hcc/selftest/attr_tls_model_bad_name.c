// EXPECT-ERROR: argument must be one of
// tls_model("...") must name one of the four TLS models.
__thread int b __attribute__((tls_model("unknown")));
int main(void) { return 0; }
