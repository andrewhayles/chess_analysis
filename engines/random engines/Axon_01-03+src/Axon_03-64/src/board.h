#pragma once
#include "common.h"

class board
{
public:
	[[nodiscard]] bool lone_king() const;
	[[nodiscard]] bool recapture(uint32_t move) const;
	bool castl_rights[4];
	int half_move_cnt;
	int king_sq[2];
	int move_cnt;
	int not_turn;
	int turn;
	uint16_t capture;
	uint8_t phase;
	uint8_t piece_sq[64];
	uint64_t ep_sq;
	uint64_t key;
	uint64_t pieces[6];
	uint64_t side[3];
	void clear();
	void new_move(uint32_t move);
	void null_move(uint64_t& ep_copy, uint16_t& capt_copy);
	void parse_fen(const std::string& fen);
	void rook_moved(const uint64_t& sq64, uint16_t sq);
	void undo_null_move(const uint64_t& ep_copy, const uint16_t& capt_copy);
};
