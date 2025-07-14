#pragma once
#include "board.h"
#include "common.h"

namespace perft
{
	void root_perft(board& pos, int depth, gen_mode mode);
	uint64_t perft(board& pos, int depth, gen_mode mode);
	void reset();
	void summary();
}