#pragma once
#include "board.h"
#include "common.h"

namespace attack
{
	uint64_t add_xray(const board& pos, const uint64_t& occ, const uint64_t& set, const uint64_t& gone, uint16_t sq);
	uint64_t by_pawns(const board& pos, int col);
	uint64_t by_slider(int sl, int sq, uint64_t occ);
	uint64_t check(const board& pos, int turn, uint64_t all_sq);
	uint64_t to_square(const board& pos, uint16_t sq);
	constexpr uint64_t border[]{ file[a], file[h] };
	constexpr int cap_left[]{ 9, 55 };
	constexpr int cap_right[]{ 7, 57 };
}