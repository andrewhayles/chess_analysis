#include "attack.h"

#include "bitops.h"
#include "magic.h"
#include "movegen.h"

uint64_t attack::add_xray(const board& pos, const uint64_t& occ, const uint64_t& set, const uint64_t& gone,
	const uint16_t sq)
{
	uint64_t att{ by_slider(ROOK, sq, occ) & (pos.pieces[Rook] | pos.pieces[Queen]) };
	att |= by_slider(BISHOP, sq, occ) & (pos.pieces[Bishop] | pos.pieces[Queen]);
	att &= ~gone;

	assert(!((1ULL << sq) & att));
	assert(popcnt((att & set) ^ att) <= 1);

	return (att & set) ^ att;
}

uint64_t attack::by_pawns(const board& pos, const int col)
{
	assert(col == WHITE || col == BLACK);

	return shift(pos.pieces[Pawn] & pos.side[col] & ~border[col], cap_left[col])
		| shift(pos.pieces[Pawn] & pos.side[col] & ~border[col ^ 1], cap_right[col]);
}

uint64_t attack::by_slider(const int sl, const int sq, uint64_t occ)
{
	assert(sq >= 0 && sq < 64);
	assert(sl == ROOK || sl == BISHOP);

	occ &= magic::slider[sl][sq].mask;
	occ *= magic::slider[sl][sq].magic;
	occ >>= magic::slider[sl][sq].shift;
	return magic::attack_table[magic::slider[sl][sq].offset + static_cast<uint32_t>(occ)];
}

uint64_t attack::check(const board& pos, const int turn, uint64_t all_sq)
{
	assert(turn == WHITE || turn == BLACK);

	const uint64_t king{ pos.side[turn] & pos.pieces[King] };
	uint64_t inquire{ all_sq };

	while (inquire)
	{
		const auto sq{ lsb(inquire) };
		const uint64_t sq64{ 1ULL << sq };
		const uint64_t in_front[]{ ~(sq64 - 1), sq64 - 1 };

		uint64_t att{ by_slider(ROOK, sq, pos.side[both] & ~king) & (pos.pieces[Rook] | pos.pieces[Queen]) };
		att |= by_slider(BISHOP, sq, pos.side[both] & ~king) & (pos.pieces[Bishop] | pos.pieces[Queen]);
		att |= movegen::knight_table[sq] & pos.pieces[Knight];
		att |= movegen::king_table[sq] & pos.pieces[King];
		att |= movegen::king_table[sq] & pos.pieces[Pawn] & movegen::slide_ray[BISHOP][sq] & in_front[turn];
		att &= pos.side[turn ^ 1];

		if (att)
			all_sq ^= sq64;

		inquire &= inquire - 1;
	}
	return all_sq;
}

uint64_t attack::to_square(const board& pos, const uint16_t sq)
{
	const uint64_t in_front[]{ (1ULL << sq) - 1, ~((1ULL << sq) - 1) };

	uint64_t att{ by_slider(ROOK, sq, pos.side[both]) & (pos.pieces[Rook] | pos.pieces[Queen]) };
	att |= by_slider(BISHOP, sq, pos.side[both]) & (pos.pieces[Bishop] | pos.pieces[Queen]);
	att |= movegen::knight_table[sq] & pos.pieces[Knight];
	att |= movegen::king_table[sq] & pos.pieces[King];
	att |= movegen::king_table[sq] & pos.pieces[Pawn] & movegen::slide_ray[BISHOP][sq] & pos.side[black] &
		in_front
		[black];
	att |= movegen::king_table[sq] & pos.pieces[Pawn] & movegen::slide_ray[BISHOP][sq] & pos.side[white] &
		in_front
		[white];

	return att;
}

