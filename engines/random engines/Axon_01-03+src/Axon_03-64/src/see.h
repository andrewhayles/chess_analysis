#pragma once
#include "board.h"
#include "common.h"

namespace see
{
	int eval(const board& pos, uint32_t move);
	uint64_t lvp(const board& pos, const uint64_t& set, int col, int& piece);

	constexpr int exact_value[8]
	{
		100, 325, 325, 500, 975, 10000, 100, 0
	};
}
