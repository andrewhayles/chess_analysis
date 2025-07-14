#pragma once
#include "board.h"
#include "common.h"

class movegen
{
public:
	board& pos;
	gen_mode mode;
	uint32_t hash_move{};

	explicit movegen(board& basic_board) : pos(basic_board), mode()
	{
	}

	movegen(board& basic_board, const gen_mode genmode) : pos(basic_board), mode(genmode)
	{
		init_list();
	}

	movegen(board& basic_board, const gen_mode genmode, const uint32_t tt_move) : pos(basic_board), mode(genmode),
		hash_move(tt_move)
	{
		init_list();
	}

	[[nodiscard]] bool is_pseudolegal(uint32_t move) const;

	uint32_t movelist[lim::movegen]{};

	struct count
	{
		int capture{ 0 };
		int hash{ 0 };
		int loosing{ 0 };
		int moves{ 0 };
		int promo{ 0 };
		int quiet{ 0 };
	} cnt;

	bool in_list(uint32_t move);
	int gen_hash();
	int gen_losing();
	int gen_quiet();
	int gen_tactical();
	static uint64_t king_table[64];
	static uint64_t knight_table[64];
	static uint64_t slide_ray[2][64];
	static void init_king();
	static void init_knight();
	static void init_ray(int sl);
	uint32_t* find(uint32_t move);
	void init_list();

private:
	uint64_t evasions{};
	uint64_t pin[64]{};
	void find_evasions();
	void find_pins();
	uint64_t enemies_{}, friends_{}, fr_king_{}, pawn_rank_{};
	uint64_t gentype_[2]{};
	uint64_t not_right_{}, not_left_{};
	uint8_t ahead_{}, back_{};
	void bishop(gen_stage stage);
	void king(gen_stage stage);
	void knight(gen_stage stage);
	void pawn_capt();
	void pawn_promo();
	void pawn_quiet();
	void queen(gen_stage stage);
	void rook(gen_stage stage);
};

inline uint64_t movegen::slide_ray[2][64];
inline uint64_t movegen::knight_table[64];
inline uint64_t movegen::king_table[64];

namespace move
{
	static_assert(capture == 0, "index");
	static_assert(quiet == 1, "index");

	constexpr int cap_left[]{ 9, 55, -9 };
	constexpr int cap_right[]{ 7, 57, -7 };
	constexpr int push[]{ 8, 56, -8 };
	constexpr int double_push[]{ 16, 48, -16 };
	constexpr uint64_t promo_rank{ rank[r1] | rank[r8] };
	constexpr uint64_t third_rank[]{ rank[r3], rank[r6] };
	constexpr uint64_t border[]{ file[a], file[h] };
}