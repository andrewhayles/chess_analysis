#include <cassert>
#include "hash.h"
#include "bitops.h"
#include "board.h"

constexpr char castling_char[]{ 'K', 'k', 'Q', 'q' };
constexpr int push[]{ 8, 56 };
constexpr int phase_value[]{ 0, 2, 2, 3, 7, 0 };
constexpr int reorder[]{ 0, 2, 1, 3 };

constexpr char piece_char[][6]
{
	{'P', 'N', 'B', 'R', 'Q', 'K'},
	{'p', 'n', 'b', 'r', 'q', 'k'}
};

int mirror(const uint16_t sq)
{
	return (sq & 56) - (sq & 7) + 7;
}

bool board::lone_king() const
{
	return (pieces[King] | pieces[Pawn]) == side[both];
}

bool board::recapture(const uint32_t move) const
{
	return to_sq2(move) == capture;
}

void board::clear()
{
	for (auto& p : pieces) p = 0ULL;
	for (auto& s : side) s = 0ULL;
	for (auto& p : piece_sq) p = no_piece;
	for (auto& c : castl_rights) c = false;

	move_cnt = 0;
	half_move_cnt = 0;
	ep_sq = 0ULL;
	turn = white;
	not_turn = black;
	phase = 0;
	capture = 0;
}

void board::new_move(const uint32_t move)
{
	move_cnt += 1;
	half_move_cnt += 1;
	capture = 0;

	const move_detail md{ decode(move) };

	const uint64_t sq1_64{ 1ULL << md.sq1 };
	const uint64_t sq2_64{ 1ULL << md.sq2 };

	assert(md.sq1 >= 0 && md.sq1 < 64);
	assert(md.sq2 >= 0 && md.sq2 < 64);
	assert(md.piece != NONE);
	assert(md.piece == piece_sq[md.sq1]);
	assert(md.victim == piece_sq[md.sq2] || md.flag == ENPASSANT);

	bool s_castl_rights[4]{};
	for (int i{ 0 }; i < 4; ++i)
		s_castl_rights[i] = castl_rights[i];

	rook_moved(sq2_64, static_cast<uint16_t>(md.sq2));
	rook_moved(sq1_64, static_cast<uint16_t>(md.sq1));

	if (md.victim != no_piece)
	{
		if (md.flag == enpassant)
		{
			assert(ep_sq != 0);

			const uint64_t capt{ shift(ep_sq, push[not_turn]) };
			assert(capt & pieces[PAWNS] & side[not_turn]);

			pieces[Pawn] &= ~capt;
			side[not_turn] &= ~capt;

			const uint16_t sq_old{ static_cast<uint16_t>(lsb(capt)) };

			assert(piece_sq[sq_old] == PAWNS);
			piece_sq[sq_old] = no_piece;

			capture = static_cast<uint16_t>(md.sq2);
			key ^= zobrist::rand_key[(turn << 6) + mirror(sq_old)];
		}

		else
		{
			assert(sq2_64 & side[not_turn]);
			half_move_cnt = 0;

			side[not_turn] &= ~sq2_64;
			pieces[md.victim] &= ~sq2_64;

			capture = static_cast<uint16_t>(md.sq2);
			phase -= static_cast<uint8_t>(phase_value[md.victim]);
			key ^= zobrist::rand_key[(((md.victim << 1) + turn) << 6) + mirror(static_cast<uint16_t>(md.sq2))];

			assert(phase >= 0);
		}
	}

	else if (md.piece == Pawn)
	{
		half_move_cnt = 0;
	}

	if (ep_sq)
	{
		if (const auto file_idx{ lsb(ep_sq) & 7 }; pieces[Pawn] & side[turn] & zobrist::ep_flank[not_turn][file_idx])
			key ^= zobrist::rand_key[zobrist::offset.ep + 7 - file_idx];

		ep_sq = 0ULL;
	}

	if (md.flag == doublepush)
	{
		assert(md.piece == PAWNS && md.victim == NONE);
		ep_sq = 1ULL << ((md.sq1 + md.sq2) / 2);

		if (const auto file_idx{ md.sq1 & 7 }; pieces[Pawn] & side[not_turn] & zobrist::ep_flank[turn][file_idx])
			key ^= zobrist::rand_key[zobrist::offset.ep + 7 - file_idx];
	}

	pieces[md.piece] ^= sq1_64;
	pieces[md.piece] |= sq2_64;

	side[turn] ^= sq1_64;
	side[turn] |= sq2_64;

	piece_sq[md.sq2] = static_cast<uint8_t>(md.piece);
	piece_sq[md.sq1] = no_piece;

	const int idx{ ((md.piece << 1) + not_turn) << 6 };
	key ^= zobrist::rand_key[idx + mirror(static_cast<uint16_t>(md.sq1))];
	key ^= zobrist::rand_key[idx + mirror(static_cast<uint16_t>(md.sq2))];

	if (md.flag >= 8)
	{
		if (md.flag <= 11)
		{
			switch (md.flag)
			{
			case white_short:
				pieces[Rook] ^= 0x1, side[turn] ^= 0x1;
				pieces[Rook] |= 0x4, side[turn] |= 0x4;
				piece_sq[h1] = no_piece, piece_sq[f1] = Rook;

				key ^= zobrist::rand_key[((7 - turn) << 6) + 7];
				key ^= zobrist::rand_key[((7 - turn) << 6) + 5];
				break;

			case black_short:
				pieces[Rook] ^= 0x100000000000000, side[turn] ^= 0x100000000000000;
				pieces[Rook] |= 0x400000000000000, side[turn] |= 0x400000000000000;
				piece_sq[h8] = no_piece, piece_sq[f8] = Rook;

				key ^= zobrist::rand_key[((7 - turn) << 6) + 63];
				key ^= zobrist::rand_key[((7 - turn) << 6) + 61];
				break;

			case white_long:
				pieces[Rook] ^= 0x80, side[turn] ^= 0x80;
				pieces[Rook] |= 0x10, side[turn] |= 0x10;
				piece_sq[a1] = no_piece, piece_sq[d1] = Rook;

				key ^= zobrist::rand_key[(7 - turn) << 6];
				key ^= zobrist::rand_key[((7 - turn) << 6) + 3];
				break;

			case black_long:
				pieces[Rook] ^= 0x8000000000000000, side[turn] ^= 0x8000000000000000;
				pieces[Rook] |= 0x1000000000000000, side[turn] |= 0x1000000000000000;
				piece_sq[a8] = no_piece, piece_sq[d8] = Rook;

				key ^= zobrist::rand_key[((7 - turn) << 6) + 56];
				key ^= zobrist::rand_key[((7 - turn) << 6) + 59];
				break;

			default:
				assert(false);
			}
		}

		else
		{
			const int promo_p{ md.flag - 11 };

			assert(md.flag >= 12 && md.flag <= 15);
			assert(pieces[PAWNS] & sq2_64);
			assert(piece_sq[md.sq2] == PAWNS);

			pieces[Pawn] ^= sq2_64;
			pieces[promo_p] |= sq2_64;
			piece_sq[md.sq2] = static_cast<uint8_t>(promo_p);

			const int sq_new{ mirror(static_cast<uint16_t>(md.sq2)) };
			key ^= zobrist::rand_key[(not_turn << 6) + sq_new];
			key ^= zobrist::rand_key[(((promo_p << 1) + not_turn) << 6) + sq_new];

			phase += static_cast<uint8_t>(phase_value[promo_p]);
		}
	}

	if (sq2_64 & pieces[King])
	{
		castl_rights[turn] = false;
		castl_rights[turn + 2] = false;

		king_sq[turn] = md.sq2;
	}

	for (int i{ 0 }; i < 4; ++i)
	{
		if (s_castl_rights[i] != castl_rights[i])
			key ^= zobrist::rand_key[zobrist::offset.castling + reorder[i]];
	}

	not_turn ^= 1;
	turn ^= 1;
	key ^= zobrist::is_turn[0];

	side[both] = side[white] | side[black];

	assert(turn == (not_turn ^ 1));
	assert(side[BOTH] == (side[WHITE] ^ side[BLACK]));
	assert(zobrist::to_key(*this) == key);
}

void board::null_move(uint64_t& ep_copy, uint16_t& capt_copy)
{
	key ^= zobrist::is_turn[0];
	capt_copy = capture;

	if (ep_sq)
	{
		if (const auto file_idx{ lsb(ep_sq) & 7 }; pieces[Pawn] & side[turn] & zobrist::ep_flank[not_turn][file_idx])
			key ^= zobrist::rand_key[zobrist::offset.ep + 7 - file_idx];

		ep_copy = ep_sq;
		ep_sq = 0;
	}

	half_move_cnt += 1;
	move_cnt += 1;
	turn ^= 1;
	not_turn ^= 1;

	assert(zobrist::to_key(*this) == key);
}

void board::parse_fen(const std::string& fen)
{
	clear();
	int sq{ 63 };
	uint32_t focus{ 0 };
	assert(focus < fen.size());

	while (focus < fen.size() && fen[focus] != ' ')
	{
		assert(sq >= 0);

		if (fen[focus] == '/')
		{
			focus += 1;
			assert(focus < fen.size());
			continue;
		}
		if (isdigit(fen[focus]))
		{
			sq -= fen[focus] - '0';
			assert(fen[focus] - '0' <= 8 && fen[focus] - '0' >= 1);
		}
		else
		{
			for (int piece{ Pawn }; piece <= King; ++piece)
			{
				for (int col{ white }; col <= black; ++col)
				{
					if (fen[focus] == piece_char[col][piece])
					{
						pieces[piece] |= 1ULL << sq;
						side[col] |= 1ULL << sq;
						piece_sq[sq] = static_cast<uint8_t>(piece);

						phase += static_cast<uint8_t>(phase_value[piece]);
						break;
					}
				}
			}
			sq -= 1;
		}
		focus += 1;
		assert(focus < fen.size());
	}

	side[both] = side[white] | side[black];
	assert(side[BOTH] == (side[WHITE] ^ side[BLACK]));

	for (int col{ white }; col <= black; ++col)
		king_sq[col] = lsb(pieces[King] & side[col]);

	focus += 1;
	if (fen[focus] == 'w')
		turn = white;
	else if (fen[focus] == 'b')
		turn = black;
	not_turn = turn ^ 1;

	focus += 2;
	while (focus < fen.size() && fen[focus] != ' ')
	{
		for (int i{ 0 }; i < 4; ++i)
		{
			if (fen[focus] == castling_char[i])
				castl_rights[i] = true;
		}
		focus += 1;
	}

	focus += 1;
	if (fen[focus] == '-')
		focus += 1;
	else
	{
		ep_sq = to_bb(fen.substr(focus, 2));
		focus += 2;
	}

	key = zobrist::to_key(*this);

	if (focus >= fen.size() - 1)
		return;

	focus += 1;
	std::string half_moves;
	while (focus < fen.size() && fen[focus] != ' ')
		half_moves += fen[focus++];
	half_move_cnt = stoi(half_moves);

	focus += 1;
	std::string moves;
	while (focus < fen.size() && fen[focus] != ' ')
		moves += fen[focus++];
	move_cnt = stoi(moves) * 2 - 1 - turn;
}

void board::rook_moved(const uint64_t& sq64, const uint16_t sq)
{
	if (sq64 & pieces[Rook])
	{
		if (sq64 & side[white])
		{
			if (sq == h1) castl_rights[ws] = false;
			else if (sq == a1) castl_rights[wl] = false;
		}
		else
		{
			if (sq == h8) castl_rights[bs] = false;
			else if (sq == a8) castl_rights[bl] = false;
		}
	}
}

void board::undo_null_move(const uint64_t& ep_copy, const uint16_t& capt_copy)
{
	key ^= zobrist::is_turn[0];
	capture = capt_copy;

	if (ep_copy)
	{
		if (const auto file_idx{ lsb(ep_copy) & 7 }; pieces[Pawn] & side[not_turn] & zobrist::ep_flank[turn][file_idx])
			key ^= zobrist::rand_key[zobrist::offset.ep + 7 - file_idx];

		ep_sq = ep_copy;
	}

	half_move_cnt -= 1;
	move_cnt -= 1;
	turn ^= 1;
	not_turn ^= 1;

	assert(zobrist::to_key(*this) == key);
}
