#include "movepick.h"

void movepick::gen_weight()
{
	list.cnt.moves = list.cnt.hash = list.cnt.capture = list.cnt.promo = list.cnt.quiet = 0;

	switch (stage_[cnt_.cycles])
	{
	case all:
		cnt_.moves = list.gen_tactical() + list.gen_quiet();
		weight_.tactical_see();
		weight_.quiet();
		weight_.hash(all);
		break;

	case Hash:
		cnt_.moves = list.gen_hash();
		weight_.hash(Hash);
		break;

	case tactical:
		cnt_.moves = list.gen_tactical();
		weight_.tactical();
		break;

	case winning:
		cnt_.moves = list.gen_tactical();
		weight_.tactical_see();
		weight_.hash(winning);
		break;

	case losing:
		cnt_.moves = list.gen_losing();
		weight_.losing();
		weight_.hash(losing);
		break;

	case quiet:
		cnt_.moves = list.gen_quiet();
		weight_.quiet();
		weight_.hash(quiet);
		break;

	case capture:
		break;

	default:
		assert(false);
	}

	cnt_.attempts = cnt_.moves;
}

uint32_t* movepick::next()
{
	while (cnt_.attempts == 0)
	{
		cnt_.cycles += 1;
		if (cnt_.cycles >= cnt_.max_cycles)
			return nullptr;

		gen_weight();
	}

	int best_idx{ -1 };
	uint64_t best_score{ 0 };

	for (int i{ 0 }; i < cnt_.moves; ++i)
	{
		if (weight_.score[i] > best_score)
		{
			best_score = weight_.score[i];
			best_idx = i;
		}
	}

	assert(cnt.attempts >= 1);
	assert(cnt.moves == list.cnt.moves);

	cnt_.attempts -= 1;

	if (best_idx == -1)
	{
		assert(stage[cnt.cycles] != HASH);
		assert(cnt.attempts == 0 || stage[cnt.cycles] == WINNING);

		return next();
	}

	weight_.score[best_idx] = 0ULL;

	return &list.movelist[best_idx];
}

void movepick::rearrange_root(uint64_t nodes[], const uint32_t* pv_move)
{
	cnt_.attempts = cnt_.moves;
	weight_.root_dynamic(nodes, pv_move);
}