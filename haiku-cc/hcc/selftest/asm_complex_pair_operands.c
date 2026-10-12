// EXPECT-EXIT: 12
// Extended asm: a 16-byte _Complex long long GPR operand occupies two registers
// (low and high qword); a read-write operand round-trips through both halves.
int main(void) {
    __complex__ long long v;
    __real__ v = 5;
    __imag__ v = 7;
    asm("" : "+r"(v));
    return (int)(__real__ v + __imag__ v);
}
