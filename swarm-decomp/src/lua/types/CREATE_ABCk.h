#pragma once
#include "Instruction.h"

#define CREATE_ABCk(o,a,b,c,k) (((Instruction)(o)) | (((Instruction)(a)) << 7) | (((Instruction)(b)) << 16) | (((Instruction)(c)) << 24) | (((Instruction)(k)) << 15))
