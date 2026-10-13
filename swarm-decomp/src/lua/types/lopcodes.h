#pragma once
#include "Instruction.h"

enum OpCode {
  OP_LOADK = 3,
  OP_LOADKX = 4,
  OP_EXTRAARG = 0x52
};

#define POS_OP 0
#define POS_A 7
#define POS_Bx 15
#define MAXARG_Bx ((1 << 17) - 1)
#define CREATE_ABx(o,a,bc) (((Instruction)(o)) | (((Instruction)(a)) << POS_A) | (((Instruction)(bc)) << POS_Bx))
#define CREATE_Ax(o,a) (((Instruction)(o)) | (((Instruction)(a)) << POS_A))
