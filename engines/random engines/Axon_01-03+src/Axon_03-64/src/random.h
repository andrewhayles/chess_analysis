#pragma once
#include "common.h"

class rand_xor
{
	uint64_t rand64();
	uint64_t seed_;

public:
	explicit rand_xor(const uint64_t new_seed) : seed_(new_seed)
	{
	}
	uint64_t sparse64();
};
