#pragma once

#include <cassert>
#include <iostream>
#include <string>

const std::string eng_name = "Axon";
const std::string version = "0.3";
const std::string platform = "x64";
const std::string author = "Jasper Sinclair";

enum square
{
	h1, g1, f1, e1, d1, c1, b1, a1,
	h2, g2, f2, e2, d2, c2, b2, a2,
	h3, g3, f3, e3, d3, c3, b3, a3,
	h4, g4, f4, e4, d4, c4, b4, a4,
	h5, g5, f5, e5, d5, c5, b5, a5,
	h6, g6, f6, e6, d6, c6, b6, a6,
	h7, g7, f7, e7, d7, c7, b7, a7,
	h8, g8, f8, e8, d8, c8, b8, a8
};

enum file_index
{
	h,
	g,
	f,
	e,
	d,
	c,
	b,
	a
};

enum rank_index
{
	r1,
	r2,
	r3,
	r4,
	r5,
	r6,
	r7,
	r8
};

enum piece_index
{
	Pawn = 0,
	Knight = 1,
	Bishop = 2,
	Rook = 3,
	Queen = 4,
	King = 5,
	no_piece = 7
};

enum pawn_move
{
	doublepush = 5,
	enpassant = 6
};

enum castling
{
	white_short = 8,
	black_short = 9,
	white_long = 10,
	black_long = 11
};

enum promo
{
	promo_knight = 12,
	promo_bishop = 13,
	promo_rook = 14,
	promo_queen = 15
};

enum slider_type
{
	ROOK,
	BISHOP
};

enum side_index
{
	white,
	black,
	both
};

enum gen_mode
{
	pseudo,
	legal
};

enum gen_stage
{
	capture,
	quiet,
	tactical,
	winning,
	losing,
	Hash,
	all,
};

enum game_stage
{
	mg,
	eg
};

enum castling_right
{
	ws,
	bs,
	wl,
	bl
};

enum score_type
{
	no_score = 20000,
	max_score = 10000,
	mate_score = 9000,
};

enum move_type
{
	no_move
};

enum bound_type
{
	exact = 1,
	upper = 2,
	lower = 3
};

struct move_detail
{
	int sq1;
	int sq2;
	int piece;
	int victim;
	int turn;
	uint8_t flag;
};

namespace lim
{
	constexpr uint64_t nodes{~0ULL};
	constexpr uint64_t movetime{~0ULL};

	constexpr int depth{128};

	constexpr int period{1024};
	constexpr int movegen{256};

	constexpr int hash{4096};
	constexpr int min_cont{-100};
	constexpr int max_cont{100};
}

constexpr uint64_t file[]
{
	0x0101010101010101,
	0x0202020202020202,
	0x0404040404040404,
	0x0808080808080808,
	0x1010101010101010,
	0x2020202020202020,
	0x4040404040404040,
	0x8080808080808080
};

constexpr uint64_t rank[]
{
	0xffULL,
	0xffULL << 8,
	0xffULL << 16,
	0xffULL << 24,
	0xffULL << 32,
	0xffULL << 40,
	0xffULL << 48,
	0xffULL << 56
};

namespace postfix
{
	const std::string promo[]{"n", "b", "r", "q"};
}

inline uint64_t shift(const uint64_t bb, const int shift)
{
	return (bb << shift) | (bb >> (64 - shift));
}

inline int to_sq1(const uint32_t move)
{
	return static_cast<int>(move & 0x3f);
}

inline int to_sq2(const uint32_t move)
{
	return static_cast<int>(move >> 6) & 0x3f;
}

inline uint8_t to_flag(const uint32_t move)
{
	return static_cast<uint8_t>((move >> 12) & 0xf);
}

inline int to_piece(const uint32_t move)
{
	return static_cast<int>(move >> 16) & 0x7;
}

inline int to_victim(const uint32_t move)
{
	return static_cast<int>(move >> 19) & 0x7;
}

inline int to_turn(const uint32_t move)
{
	assert(move >> 23 == 0);
	return static_cast<int>(move) >> 22;
}

inline std::string to_promo(const uint8_t flag)
{
	assert(flag > 0 && flag < 16);
	return flag >= 12 ? postfix::promo[flag - 12] : "";
}

inline int to_idx(const std::string& sq)
{
	assert(sq.size() == 2);
	return 'h' - sq.front() + ((sq.back() - '1') << 3);
}

inline uint64_t to_bb(const std::string& sq)
{
	return 1ULL << to_idx(sq);
}

inline std::string to_str(const int sq)
{
	std::string str;
	str += 'h' - static_cast<char>(sq & 7);
	str += '1' + static_cast<char>(sq >> 3);

	return str;
}

inline uint32_t encode(const uint32_t sq1, const uint32_t sq2, const int flag, const int piece, const int victim,
                       const int turn)
{
	assert(turn == (turn & 1));
	assert(piece <= 5);
	assert(victim <= 4 || victim == 7);

	return sq1 | sq2 << 6 | flag << 12 | piece << 16 | victim << 19 | turn << 22;
}

inline move_detail decode(const uint32_t move)
{
	return {to_sq1(move), to_sq2(move), to_piece(move), to_victim(move), to_turn(move), to_flag(move)};
}

inline std::string algebraic(const uint32_t move)
{
	return to_str(to_sq1(move)) + to_str(to_sq2(move)) + to_promo(to_flag(move));
}
