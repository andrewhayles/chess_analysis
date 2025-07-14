#include "see.h"

#include <algorithm>

#include "attack.h"

int see::eval(const board& pos, const uint32_t move)
{
	int gain[32]{ 0 }, d{ 0 };
	const uint64_t all_xray{ pos.pieces[Pawn] | pos.pieces[Rook] | pos.pieces[Bishop] | pos.pieces[Queen] };

	const auto sq1{ to_sq1(move) };
	const auto sq2{ to_sq2(move) };
	uint64_t sq1_64{ 1ULL << sq1 };

	uint64_t occ{ pos.side[both] };
	uint64_t set{ attack::to_square(pos, static_cast<uint16_t>(sq2)) };
	uint64_t gone{ 0ULL };
	int piece{ to_piece(move) };

	assert(to_victim(move) != NONE);
	gain[d] = exact_value[to_victim(move)];

	do
	{
		gain[++d] = exact_value[piece] - gain[d - 1];

		if (std::max(-gain[d - 1], gain[d]) < 0)
			break;

		set ^= sq1_64;
		occ ^= sq1_64;
		gone |= sq1_64;

		if (sq1_64 & all_xray)
			set |= attack::add_xray(pos, occ, set, gone, static_cast<uint16_t>(sq2));

		sq1_64 = lvp(pos, set, pos.turn ^ (d & 1), piece);
	} while (sq1_64);

	while (--d) gain[d - 1] = -std::max(-gain[d - 1], gain[d]);

	return gain[0];
}

uint64_t see::lvp(const board& pos, const uint64_t& set, const int col, int& piece)
{
	for (piece = Pawn; piece <= King; ++piece)
	{
		if (const uint64_t subset{ set & pos.pieces[piece] & pos.side[col] })
			return (subset & (subset - 1)) ^ subset;
	}
	return 0ULL;
}