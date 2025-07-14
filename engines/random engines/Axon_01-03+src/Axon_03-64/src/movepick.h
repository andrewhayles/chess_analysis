#pragma once
#include "common.h"
#include "movegen.h"
#include "sort.h"

class movepick
{
	gen_stage stage_[4]{};
	sort weight_;

	struct count
	{
		int attempts{ 0 };
		int cycles{ -1 };
		int max_cycles{ 0 };
		int moves{ 0 };
	} cnt_;

public:
	movegen list;
	board reverse;

	movepick(board& pos, uint64_t nodes[])
		: weight_(list, nodes), list(pos, legal), reverse(pos)
	{
		list.gen_tactical();
		list.gen_quiet();
		cnt_.cycles = 0;
		cnt_.max_cycles = 0;
		cnt_.attempts = cnt_.moves = list.cnt.moves;
		weight_.root_static(nodes);
	}

	movepick(board& pos, const uint32_t tt_move, sort::history_list& history, sort::killer_list& killer,
		const int depth)
		: weight_(list, history, killer, depth), list(pos, pseudo, tt_move), reverse(pos)
	{
		cnt_.max_cycles = 4;
		stage_[0] = Hash;
		stage_[1] = winning;
		stage_[2] = quiet;
		stage_[3] = losing;
	}

	explicit movepick(board& pos)
		: weight_(list), list(pos, legal), reverse(pos)
	{
		cnt_.max_cycles = 1;
		stage_[0] = tactical;
	}

	void gen_weight();
	uint32_t* next();
	void rearrange_root(uint64_t nodes[], const uint32_t* pv_move);
};
