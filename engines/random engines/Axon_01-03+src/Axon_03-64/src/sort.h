#pragma once
#include "board.h"
#include "common.h"
#include "movegen.h"
#include "sort.h"

class sort
{
public:
	struct history_list
	{
		uint64_t list[2][6][64];
	};

	struct killer_list
	{
		uint32_t list[lim::depth][2];
	};

	uint64_t score[lim::movegen]{};

private:
	movegen& list_;

	struct main_parameters
	{
		int depth;
		killer_list* killer;
		history_list* history;
	} main_{};

	static bool losing(const board& pos, uint32_t move);
	static int mvv_lva(uint32_t move);
	static int mvv_lva_promo(uint32_t move);

public:
	sort(movegen& movelist, uint64_t[]) : list_(movelist)
	{
	}

	sort(movegen& movelist, history_list& history, killer_list& killer, const int depth) : list_(movelist)
	{
		main_.depth = depth;
		main_.history = &history;
		main_.killer = &killer;
	}

	explicit sort(movegen& movelist) : list_(movelist)
	{
	}

	void hash(gen_stage stage);
	void losing();
	void quiet();
	void root_dynamic(uint64_t nodes[], const uint32_t* pv_move);
	void root_static(uint64_t nodes[]);
	void tactical_see();
	void tactical();
};

namespace value
{
	constexpr int pvalue[]{ 1, 3, 3, 5, 9, 0, 1 };
	constexpr uint64_t maxscore{ 1ULL << 63 };
	constexpr uint64_t capt_score{ 1ULL << 62 };
}