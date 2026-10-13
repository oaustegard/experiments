#pragma once
#include "Instruction.h"
#include "luaP_opmodes.h"

/* opcode numbers of Lua 5.4 lopcodes.h used by the ldebug TU */
#define DOP_MOVE 0
#define DOP_LOADNIL 8
#define DOP_GETUPVAL 9
#define DOP_LOADK 3
#define DOP_LOADKX 4
#define DOP_GETTABUP 11
#define DOP_GETTABLE 12
#define DOP_GETI 13
#define DOP_GETFIELD 14
#define DOP_SELF 20
#define DOP_CALL 68
#define DOP_TAILCALL 69
#define DOP_JMP 56
#define DOP_TFORCALL 76

#define DGET_OPCODE(i) (((Instruction)(i)) & 0x7f)
#define DGETARG_A(i) ((((Instruction)(i)) >> 7) & 0xff)
#define DGETARG_B(i) ((((Instruction)(i)) >> 16) & 0xff)
#define DGETARG_C(i) ((((Instruction)(i)) >> 24) & 0xff)
#define DGETARG_k(i) ((((Instruction)(i)) >> 15) & 1)
#define DGETARG_Bx(i) (((Instruction)(i)) >> 15)
#define DGETARG_Ax(i) (((Instruction)(i)) >> 7)
#define DGETARG_sJ(i) ((int)(((Instruction)(i)) >> 7) - 0xffffff)
#define DtestAMode(op) (luaP_opmodes[op] & (1 << 3))
#define DtestMMMode(op) (luaP_opmodes[op] & (1 << 7))
#define DOP_SETTABUP 15
#define DOP_SETTABLE 16
#define DOP_SETI 17
#define DOP_SETFIELD 18
#define DOP_UNM 49
#define DOP_BNOT 50
#define DOP_LEN 52
#define DOP_CONCAT 53
#define DOP_CLOSE 54
#define DOP_EQ 57
#define DOP_LT 58
#define DOP_LE 59
#define DOP_EQK 60
#define DOP_EQI 61
#define DOP_LTI 62
#define DOP_LEI 63
#define DOP_GTI 64
#define DOP_GEI 65
#define DOP_RETURN 70
#define DOP_MMBIN 46
#define DOP_MMBINI 47
#define DOP_MMBINK 48
