#include "movegen.h"

#include <algorithm>

#include "attack.h"
#include "bitops.h"
#include "magic.h"

void movegen::init_knight()
{
	constexpr magic::pattern jump[]
	{
		{15, 0xffff010101010101}, {6, 0xff03030303030303},
		{54, 0x03030303030303ff}, {47, 0x010101010101ffff},
		{49, 0x808080808080ffff}, {58, 0xc0c0c0c0c0c0c0ff},
		{10, 0xffc0c0c0c0c0c0c0}, {17, 0xffff808080808080}
	};

	for (int sq{ 0 }; sq < 64; ++sq)
	{
		const uint64_t sq64{ 1ULL << sq };

		for (int dir{ 0 }; dir < 8; ++dir)
		{
			if (const uint64_t att{ sq64 }; !(att & jump[dir].border))
				knight_table[sq] |= shift(att, jump[dir].shift);
		}
	}
}

void movegen::init_king()
{
	for (int sq{ 0 }; sq < 64; ++sq)
	{
		const uint64_t sq64{ 1ULL << sq };

		for (int dir{ 0 }; dir < 8; ++dir)
		{
			if (const uint64_t att{ sq64 }; !(att & magic::ray[dir].border))
				king_table[sq] |= shift(att, magic::ray[dir].shift);
		}
	}
}

void movegen::init_list()
{
	ahead_ = static_cast<uint8_t>(pos.turn);
	back_ = static_cast<uint8_t>(pos.turn << 1);
	not_right_ = ~move::border[pos.not_turn];
	not_left_ = ~move::border[pos.turn];
	pawn_rank_ = move::third_rank[pos.turn];
	friends_ = pos.side[pos.turn];
	enemies_ = pos.side[pos.not_turn];
	fr_king_ = pos.pieces[King] & friends_;
	gentype_[capture] = enemies_;
	gentype_[quiet] = ~pos.side[both];

	if (mode == legal)
	{
		find_pins();
		find_evasions();
	}
	else
	{
		assert(mode == PSEUDO);
		evasions = ~0ULL;
	}
}
void real_shift(uint64_t& bb, const int shift)
{
	bb = (bb << shift) | (bb >> (64 - shift));
}

int movegen::gen_losing()
{
	for (int i{ 0 }; i < cnt.loosing; ++i)
	{
		movelist[i] = movelist[lim::movegen - 1 - i];
	}

	assert(cnt.moves == 0);
	assert(cnt.quiet == 0);
	assert(cnt.capture == 0);
	assert(cnt.hash == 0);
	assert(cnt.promo == 0);

	cnt.moves = cnt.loosing;
	return cnt.moves;
}

int movegen::gen_hash()
{
	assert(cnt.moves == 0);
	assert(cnt.hash == 0);

	if (hash_move != no_move && is_pseudolegal(hash_move))
	{
		movelist[0] = hash_move;
		cnt.moves = cnt.hash = 1;
	}

	return cnt.moves;
}

int movegen::gen_tactical()
{
	assert(cnt.moves == 0);
	assert(cnt.capture == 0);

	king(capture);
	pawn_capt();
	queen(capture);
	rook(capture);
	bishop(capture);
	knight(capture);
	cnt.capture = cnt.moves;
	pawn_promo();
	cnt.promo = cnt.moves - cnt.capture;
	return cnt.moves;
}

int movegen::gen_quiet()
{
	assert(cnt.capture + cnt.promo == cnt.moves);
	assert(cnt.quiet == 0);

	king(quiet);
	queen(quiet);
	rook(quiet);
	bishop(quiet);
	knight(quiet);
	pawn_quiet();
	cnt.quiet = cnt.moves - cnt.capture - cnt.promo;
	return cnt.quiet;
}

void movegen::pawn_quiet()
{
	const uint64_t pushes{ shift(pos.pieces[Pawn] & friends_, move::push[ahead_]) & ~pos.side[both] & ~move::promo_rank };
	uint64_t targets{ pushes & evasions };

	while (targets)
	{
		const auto target{ lsb(targets) };
		const auto origin{ target - move::push[back_] };

		assert((1ULL << origin) & pos.pieces[PAWNS] & friends);

		if ((1ULL << target) & ~pin[origin])
			movelist[cnt.moves++] = encode(origin, target, no_piece, Pawn, no_piece, pos.turn);

		targets &= targets - 1;
	}

	uint64_t targets2x{ shift(pushes & pawn_rank_, move::push[ahead_]) & evasions & ~pos.side[both] };
	while (targets2x)
	{
		const auto target{ lsb(targets2x) };
		const auto origin{ target - move::double_push[back_] };

		assert((1ULL << origin) & pos.pieces[PAWNS] & friends);

		if ((1ULL << target) & ~pin[origin])
			movelist[cnt.moves++] = encode(origin, target, doublepush, Pawn, no_piece, pos.turn);

		targets2x &= targets2x - 1;
	}
}

void movegen::pawn_capt()
{
	uint64_t targets{ shift(pos.pieces[Pawn] & friends_ & not_left_, move::cap_left[ahead_]) & ~move::promo_rank };
	uint64_t targets_cap{ targets & enemies_ & evasions };

	while (targets_cap)
	{
		const auto target{ lsb(targets_cap) };
		const auto origin{ target - move::cap_left[back_] };

		assert((1ULL << origin) & pos.pieces[PAWNS] & friends);

		if ((1ULL << target) & ~pin[origin])
			movelist[cnt.moves++] = encode(origin, target, no_piece, Pawn, pos.piece_sq[target], pos.turn);

		targets_cap &= targets_cap - 1;
	}

	uint64_t target_ep{ targets & pos.ep_sq & shift(evasions, move::push[ahead_]) };
	if (target_ep)
	{
		const auto target{ lsb(target_ep) };
		const auto origin{ target - move::cap_left[back_] };

		assert((1ULL << origin) & pos.pieces[PAWNS] & friends);

		target_ep &= ~pin[origin];
		if (target_ep)
			movelist[cnt.moves++] = encode(origin, target, enpassant, Pawn, Pawn, pos.turn);
	}

	targets = shift(pos.pieces[Pawn] & friends_ & not_right_, move::cap_right[ahead_]) & ~move::promo_rank;
	targets_cap = targets & enemies_ & evasions;

	while (targets_cap)
	{
		const auto target{ lsb(targets_cap) };
		const auto origin{ target - move::cap_right[back_] };

		assert((1ULL << origin) & pos.pieces[PAWNS] & friends);

		if ((1ULL << target) & ~pin[origin])
			movelist[cnt.moves++] = encode(origin, target, no_piece, Pawn, pos.piece_sq[target], pos.turn);

		targets_cap &= targets_cap - 1;
	}

	target_ep = targets & pos.ep_sq & shift(evasions, move::push[ahead_]);
	if (target_ep)
	{
		const auto target{ lsb(target_ep) };
		const auto origin{ target - move::cap_right[back_] };

		assert((1ULL << origin) & pos.pieces[PAWNS] & friends);

		target_ep &= ~pin[origin];
		if (target_ep)
			movelist[cnt.moves++] = encode(origin, target, enpassant, Pawn, Pawn, pos.turn);
	}
}

void movegen::pawn_promo()
{
	uint64_t targets{
		shift(pos.pieces[Pawn] & friends_ & not_left_, move::cap_left[ahead_]) & evasions & move::promo_rank & enemies_
	};
	while (targets)
	{
		const auto target{ lsb(targets) };
		const auto origin{ target - move::cap_left[back_] };

		assert((1ULL << origin) & pos.pieces[PAWNS] & friends);

		if ((1ULL << target) & ~pin[origin])
		{
			for (int flag{ 15 }; flag >= 12; --flag)
				movelist[cnt.moves++] = encode(origin, target, flag, Pawn, pos.piece_sq[target], pos.turn);
		}

		targets &= targets - 1;
	}

	targets = shift(pos.pieces[Pawn] & friends_ & not_right_, move::cap_right[ahead_]) & evasions & move::promo_rank & enemies_;
	while (targets)
	{
		const auto target{ lsb(targets) };
		const auto origin{ target - move::cap_right[back_] };

		assert((1ULL << origin) & pos.pieces[PAWNS] & friends);

		if ((1ULL << target) & ~pin[origin])
		{
			for (int flag{ 15 }; flag >= 12; --flag)
				movelist[cnt.moves++] = encode(origin, target, flag, Pawn, pos.piece_sq[target], pos.turn);
		}

		targets &= targets - 1;
	}

	targets = shift(pos.pieces[Pawn] & friends_, move::push[ahead_]) & ~pos.side[both] & evasions & move::promo_rank;
	while (targets)
	{
		const auto target{ lsb(targets) };
		const auto origin{ target - move::push[back_] };

		assert((1ULL << origin) & pos.pieces[PAWNS] & friends);

		if ((1ULL << target) & ~pin[origin])
		{
			for (int flag{ 15 }; flag >= 12; --flag)
				movelist[cnt.moves++] = encode(origin, target, flag, Pawn, no_piece, pos.turn);
		}

		targets &= targets - 1;
	}
}

void movegen::bishop(const gen_stage stage)
{
	uint64_t pieces{ pos.pieces[Bishop] & friends_ };
	while (pieces)
	{
		const auto sq1{ lsb(pieces) };
		uint64_t targets{ attack::by_slider(BISHOP, sq1, pos.side[both]) & gentype_[stage] & evasions & ~pin[sq1] };

		while (targets)
		{
			const auto sq2{ lsb(targets) };
			movelist[cnt.moves++] = encode(sq1, sq2, no_piece, Bishop, pos.piece_sq[sq2], pos.turn);
			targets &= targets - 1;
		}
		pieces &= pieces - 1;
	}
}

void movegen::king(const gen_stage stage)
{
	uint64_t targets{ attack::check(pos, pos.turn, king_table[pos.king_sq[pos.turn]] & gentype_[stage]) };
	while (targets)
	{
		const auto sq{ lsb(targets) };
		movelist[cnt.moves++] = encode(pos.king_sq[pos.turn], sq, no_piece, King, pos.piece_sq[sq], pos.turn);
		targets &= targets - 1;
	}

	if (stage == quiet && fr_king_ & 0x800000000000008)
	{
		const uint64_t rank_king{ rank[pos.turn * 7] };

		if (pos.castl_rights[pos.turn]
			&& !(pos.side[both] & 0x0600000000000006 & rank_king)
			&& popcnt(attack::check(pos, pos.turn, 0x0e0000000000000e & rank_king)) == 3)
		{
			constexpr uint32_t target[]{ 1, 57 };
			movelist[cnt.moves++] = encode(pos.king_sq[pos.turn], target[pos.turn], (white_short + pos.turn),
				King, no_piece, pos.turn);
		}
		if (pos.castl_rights[pos.turn + 2]
			&& !(pos.side[both] & 0x7000000000000070 & rank_king)
			&& popcnt(attack::check(pos, pos.turn, 0x3800000000000038 & rank_king)) == 3)
		{
			constexpr uint32_t target[]{ 5, 61 };
			movelist[cnt.moves++] = encode(pos.king_sq[pos.turn], target[pos.turn], (white_long + pos.turn),
				King, no_piece, pos.turn);
		}
	}
}

void movegen::knight(const gen_stage stage)
{
	uint64_t pieces{ pos.pieces[Knight] & friends_ };
	while (pieces)
	{
		const auto sq1{ lsb(pieces) };
		uint64_t targets{ knight_table[sq1] & gentype_[stage] & evasions & ~pin[sq1] };

		while (targets)
		{
			const auto sq2{ lsb(targets) };
			movelist[cnt.moves++] = encode(sq1, sq2, no_piece, Knight, pos.piece_sq[sq2], pos.turn);
			targets &= targets - 1;
		}
		pieces &= pieces - 1;
	}
}

void movegen::queen(const gen_stage stage)
{
	uint64_t pieces{ pos.pieces[Queen] & friends_ };
	while (pieces)
	{
		const auto sq1{ lsb(pieces) };

		uint64_t targets{
			attack::by_slider(BISHOP, sq1, pos.side[both]) | attack::by_slider(ROOK, sq1, pos.side[both])
		};
		targets &= gentype_[stage] & evasions & ~pin[sq1];

		while (targets)
		{
			const auto sq2{ lsb(targets) };
			movelist[cnt.moves++] = encode(sq1, sq2, no_piece, Queen, pos.piece_sq[sq2], pos.turn);
			targets &= targets - 1;
		}
		pieces &= pieces - 1;
	}
}

void movegen::rook(const gen_stage stage)
{
	uint64_t pieces{ pos.pieces[Rook] & friends_ };
	while (pieces)
	{
		const auto sq1{ lsb(pieces) };
		uint64_t targets{ attack::by_slider(ROOK, sq1, pos.side[both]) & gentype_[stage] & evasions & ~pin[sq1] };

		while (targets)
		{
			const auto sq2{ lsb(targets) };
			movelist[cnt.moves++] = encode(sq1, sq2, no_piece, Rook, pos.piece_sq[sq2], pos.turn);
			targets &= targets - 1;
		}
		pieces &= pieces - 1;
	}
}

void movegen::find_pins()
{
	const auto king_sq{ pos.king_sq[pos.turn] };

	uint64_t att{ slide_ray[ROOK][king_sq] & enemies_ & (pos.pieces[Rook] | pos.pieces[Queen]) };
	att |= slide_ray[BISHOP][king_sq] & enemies_ & (pos.pieces[Bishop] | pos.pieces[Queen]);

	while (att)
	{
		uint64_t ray_to_att{ attack::by_slider(ROOK, king_sq, att) };
		ray_to_att |= attack::by_slider(BISHOP, king_sq, att);

		const uint64_t attacker{ 1ULL << lsb(att) };

		if (!(attacker & ray_to_att))
		{
			att &= att - 1;
			continue;
		}

		assert(fr_king);

		uint64_t x_ray{ 0 };
		for (int dir{ 0 }; dir < 8; ++dir)
		{
			auto flood{ fr_king_ };
			for (; !(flood & magic::ray[dir].border); flood |= shift(flood, magic::ray[dir].shift));

			if (flood & attacker)
			{
				x_ray = flood & ray_to_att;
				break;
			}
		}

		assert(x_ray & attacker);
		assert(!(x_ray & fr_king));

		if ((x_ray & friends_) && popcnt(x_ray & pos.side[both]) == 2)
		{
			assert(popcnt(x_ray & friends) == 1);

			const auto sq{ lsb(x_ray & friends_) };
			pin[sq] = ~x_ray;
		}

		else if (pos.ep_sq
			&& x_ray & friends_ & pos.pieces[Pawn]
			&& x_ray & enemies_ & pos.pieces[Pawn]
			&& popcnt(x_ray & pos.side[both]) == 3)
		{
			assert(popcnt(x_ray & enemies) == 2);

			const uint64_t enemy_pawn{ x_ray & enemies_ & pos.pieces[Pawn] };

			if (const uint64_t friend_pawn{ x_ray & friends_ & pos.pieces[Pawn] }; friend_pawn << 1 == enemy_pawn ||
				friend_pawn >> 1 == enemy_pawn)
			{
				if (pos.ep_sq == shift(enemy_pawn, move::push[pos.turn]))
				{
					const auto sq{ lsb(x_ray & friends_) };
					pin[sq] = pos.ep_sq;
				}
			}
		}

		att &= att - 1;
	}
}

void movegen::find_evasions()
{
	const uint64_t in_front[]{ ~(fr_king_ - 1), fr_king_ - 1 };
	const auto king_sq{ pos.king_sq[pos.turn] };
	assert(fr_king != 0ULL);

	uint64_t att{ attack::by_slider(ROOK, king_sq, pos.side[both]) & (pos.pieces[Rook] | pos.pieces[Queen]) };
	att |= attack::by_slider(BISHOP, king_sq, pos.side[both]) & (pos.pieces[Bishop] | pos.pieces[Queen]);
	att |= knight_table[king_sq] & pos.pieces[Knight];
	att |= king_table[king_sq] & pos.pieces[Pawn] & slide_ray[BISHOP][king_sq] & in_front[pos.turn];
	att &= enemies_;

	if (const int nr_att{ popcnt(att) }; nr_att == 0)
	{
		evasions = ~0ULL;
	}
	else if (nr_att == 1)
	{
		if (att & pos.pieces[Knight] || att & pos.pieces[Pawn])
		{
			evasions = att;
		}
		else
		{
			assert(att & pos.pieces[ROOKS] || att & pos.pieces[BISHOPS] || att & pos.pieces[QUEENS]);
			const auto every_att{
				attack::by_slider(ROOK, king_sq, pos.side[both]) | attack::by_slider(
					BISHOP, king_sq, pos.side[both])
			};

			for (int dir{ 0 }; dir < 8; ++dir)
			{
				auto flood{ fr_king_ };
				for (; !(flood & magic::ray[dir].border); flood |= shift(flood, magic::ray[dir].shift));

				if (flood & att)
				{
					evasions = flood & every_att;
					break;
				}
			}
		}
	}
	else
	{
		assert(nr_att == 2);
		evasions = 0ULL;
	}
}

bool movegen::is_pseudolegal(const uint32_t move) const
{
	const auto [sq1, sq2, piece, victim, turn, flag] {decode(move)};

	const uint64_t sq1_64{ 1ULL << sq1 };
	const uint64_t sq2_64{ 1ULL << sq2 };

	assert(md.turn == pos.turn);

	if (piece != pos.piece_sq[sq1] || (pos.side[turn ^ 1] & sq1_64))
		return false;
	if ((victim != pos.piece_sq[sq2] || (pos.side[turn] & sq2_64)) && flag != enpassant)
		return false;

	switch (piece)
	{
	case Pawn:
	{
		if (flag == doublepush)
			return pos.piece_sq[(sq1 + sq2) / 2] == no_piece;
		if (flag == enpassant)
			return pos.ep_sq == sq2_64;
		return true;
	}

	case Knight:
		return true;

	case Bishop:
		return (attack::by_slider(BISHOP, sq1, pos.side[both]) & sq2_64) != 0ULL;

	case Rook:
		return (attack::by_slider(ROOK, sq1, pos.side[both]) & sq2_64) != 0ULL;

	case Queen:
		return ((attack::by_slider(ROOK, sq1, pos.side[both]) | attack::by_slider(BISHOP, sq1, pos.side[both])) &
			sq2_64) != 0ULL;

	case King:
	{
		if (flag == no_piece)
			return true;
		assert(md.flag >= CASTLING::WHITE_SHORT && md.flag <= CASTLING::BLACK_LONG);

		if (!pos.castl_rights[flag - 8])
			return false;
		if (popcnt(attack::check(pos, turn, (1ULL << ((sq1 + sq2) / 2)) | sq1_64)) != 2)
			return false;

		if (flag <= black_short && (pos.side[both] & 0x0600000000000006 & rank[turn * 7]))
			return false;
		if (flag >= white_long && (pos.side[both] & 0x7000000000000070 & rank[turn * 7]))
			return false;

		return true;
	}

	default:
		assert(false);
		return false;
	}
}

uint32_t* movegen::find(const uint32_t move)
{
	return std::find(movelist, movelist + cnt.moves, move);
}

bool movegen::in_list(const uint32_t move)
{
	return find(move) != movelist + cnt.moves;
}

void movegen::init_ray(const int sl)
{
	assert(sl == ROOK || sl == BISHOP);

	for (int sq{ 0 }; sq < 64; ++sq)
	{
		const uint64_t sq64{ 1ULL << sq };

		for (int dir{ sl }; dir < 8; dir += 2)
		{
			uint64_t flood{ sq64 };
			while (!(flood & magic::ray[dir].border))
			{
				real_shift(flood, magic::ray[dir].shift);
				slide_ray[sl][sq] |= flood;
			}
		}
	}
}
