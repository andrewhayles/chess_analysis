#include "sort.h"

#include "eval.h"
#include "see.h"

int sort::mvv_lva(const uint32_t move)
{
	assert(to_victim(move) != NONE);

	return see::exact_value[to_victim(move)] - value::pvalue[to_piece(move)];
}

bool sort::losing(const board& pos, const uint32_t move)
{
	assert(to_victim(move) != NONE);
	assert(to_piece(move) != NONE);

	const auto piece{ to_piece(move) };

	if (piece == King)
		return false;
	if (value::pvalue[to_victim(move)] >= value::pvalue[piece])
		return false;
	return see::eval(pos, move) < 0;
}

int sort::mvv_lva_promo(const uint32_t move)
{
	assert(to_flag(move) >= PROMO_KNIGHT);
	assert(see::exact_value[NONE] == 0);

	const int victim{ see::exact_value[to_victim(move)] + see::exact_value[to_flag(move) - 11] };

	return victim - 2 * see::exact_value[Pawn] - value::pvalue[to_flag(move) - 11];
}

void sort::hash(const gen_stage stage)
{
	if (list_.hash_move != no_move)
	{
		const uint32_t* pos_list{ list_.find(list_.hash_move) };

		assert(stage != ALL);

		if (pos_list != list_.movelist + list_.cnt.moves)
			score[pos_list - list_.movelist] = (stage == Hash ? value::maxscore : 0ULL);
	}
}

void sort::losing()
{
	for (int i{ 0 }; i < list_.cnt.loosing; ++i)
	{
		score[i] = mvv_lva(list_.movelist[i]);
	}
}

void sort::quiet()
{
	for (int i{ list_.cnt.capture + list_.cnt.promo }; i < list_.cnt.capture + list_.cnt.promo + list_.cnt.quiet; ++i)
	{
		assert(list.pos.turn == to_turn(list.movelist[i]));
		score[i] = main_.history->list[list_.pos.turn][to_piece(list_.movelist[i])][to_sq2(list_.movelist[i])];
	}

	for (int slot{ 0 }; slot < 2; ++slot)
	{
		if (const auto piece{ to_piece(main_.killer->list[main_.depth][slot]) }; piece != list_.pos.piece_sq[to_sq1(
			main_.killer->list[main_.depth][slot])])
			continue;

		for (int i{ list_.cnt.capture + list_.cnt.promo }; i < list_.cnt.capture + list_.cnt.promo + list_.cnt.quiet; ++i)
		{
			if (list_.movelist[i] == main_.killer->list[main_.depth][slot])
			{
				score[i] = value::capt_score + 2 - slot;
				break;
			}
		}
	}
}

void sort::root_dynamic(uint64_t nodes[], const uint32_t* pv_move)
{
	assert(*pv_move != NO_MOVE);

	for (int i{ 0 }; i < list_.cnt.moves; ++i)
	{
		score[i] = nodes[i];
		nodes[i] >>= 1;
	}

	score[pv_move - list_.movelist] = value::maxscore;
}

void sort::root_static(uint64_t nodes[])
{
	const board saved{ list_.pos };

	for (int i{ 0 }; i < list_.cnt.capture; ++i)
	{
		list_.pos.new_move(list_.movelist[i]);
		score[i] = static_cast<uint64_t>(no_score) - eval::static_eval(list_.pos) + see::eval(saved, list_.movelist[i]);
		list_.pos = saved;
	}

	for (int i{ list_.cnt.capture }; i < list_.cnt.moves; ++i)
	{
		list_.pos.new_move(list_.movelist[i]);
		score[i] = static_cast<uint64_t>(no_score) - eval::static_eval(list_.pos);
		list_.pos = saved;
	}

	for (int i{ 0 }; i < list_.cnt.moves; ++i)
		nodes[i] = score[i];
}

void sort::tactical()
{
	for (int i{ 0 }; i < list_.cnt.capture; ++i)
		score[i] = value::capt_score + mvv_lva(list_.movelist[i]);

	for (int i{ list_.cnt.capture }; i < list_.cnt.capture + list_.cnt.promo; ++i)
		score[i] = value::capt_score + mvv_lva_promo(list_.movelist[i]);
}

void sort::tactical_see()
{
	assert(list.cnt.loosing == 0);

	for (int i{ 0 }; i < list_.cnt.capture; ++i)
	{
		if (losing(list_.pos, list_.movelist[i]))
		{
			assert(stage != ALL);

			score[i] = 0ULL;
			list_.movelist[lim::movegen - ++list_.cnt.loosing] = list_.movelist[i];
		}

		else
			score[i] = value::capt_score + mvv_lva(list_.movelist[i]);
	}

	for (int i{ list_.cnt.capture }; i < list_.cnt.capture + list_.cnt.promo; ++i)
		score[i] = value::capt_score + mvv_lva_promo(list_.movelist[i]);
}