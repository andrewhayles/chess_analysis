#include "random.h"

uint64_t rand_xor::rand64()
{
	seed_ ^= seed_ >> 12;
	seed_ ^= seed_ << 25;
	seed_ ^= seed_ >> 27;
	return seed_ * 0x2545f4914f6cdd1dULL;
}

uint64_t rand_xor::sparse64()
{
	return rand64() & rand64() & rand64();
}
