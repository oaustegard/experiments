#pragma once
/* opcode numbers used by lcode.c (lopcodes.h, Lua 5.4) */
typedef int OpCode;
enum {
  OP_RETURN0 = 70, OP_RETURN1 = 72, OP_RETURN = 71, OP_JMP = 56,
  OP_TEST = 66, OP_TESTSET = 67
};
#define NO_JUMP (-1)
#define NO_REG 0xff
#define OFFSET_sJ 0xffffff
#define MAXARG_sJ 0x1ffffff
