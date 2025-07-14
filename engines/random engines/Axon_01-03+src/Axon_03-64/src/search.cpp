#include "search.h"

#include "attack.h"
#include "engine.h"
#include "eval.h"
#include "hash.h"
#include "see.h"
#include "sort.h"

int search::alphabeta(board& pos, const int ply, const int depth, int beta, int alpha)
{
	assert(beta > alpha);
	assert(ply >= 0 && ply < lim::depth);

	const bool pv_node = beta != alpha + 1;

	if (verify(pos, depth))
		return contempt[pos.turn];

	if (ply == 0 || depth >= lim::depth)
		return qsearch(pos, alpha, beta);

	if (stop_thread())
	{
		engine::stop = true;
		return no_score;
	}

	if (max_score - depth < beta)
	{
		beta = max_score - depth;
		if (beta <= alpha)
			return alpha;
	}

	int tt_score{ no_score };
	uint32_t tt_move{ no_move };
	uint8_t tt_flag{ 0 };

	if (hash::probe(pos, tt_move, tt_score, ply, depth, tt_flag))
	{
		assert(tt_score != NO_SCORE);
		assert(tt_flag != 0);

		if (pv_node)
		{
			if (tt_score <= alpha || tt_score >= beta)
				tt_move = no_move;
		}
		else if (tt_score >= beta || tt_score <= alpha)
		{
			if (tt_flag == lower && tt_score >= beta) return beta;
			if (tt_flag == upper && tt_score <= alpha) return alpha;
			if (tt_flag == exact) return tt_score;
		}
	}

	if (tt_flag != exact)
		tt_move = no_move;

	const bool in_check{ is_check(pos) };
	const bool skip_pruning{ pv_node || in_check || no_pruning[depth] };

	int score{ pv_node || in_check ? no_score : eval::static_eval(pos) };

	if (ply <= 3
		&& !is_mate(beta)
		&& !skip_pruning
		&& score - ply * 50 >= beta)
	{
		assert(score != NO_SCORE);
		return beta;
	}

	if (ply <= 3
		&& !skip_pruning
		&& score + ply * 50 + 100 <= alpha)
	{
		const auto raz_alpha{ alpha - ply * 50 - 100 };
		const auto new_score{ qsearch(pos, raz_alpha, raz_alpha + 1) };

		if (engine::stop)
			return no_score;

		if (new_score <= raz_alpha)
			return alpha;
	}

	if (ply >= 3
		&& !skip_pruning
		&& !pos.lone_king()
		&& score >= beta)
	{
		constexpr int R{ 2 };
		uint64_t ep_copy{ 0 };
		uint16_t capt_copy{ 0 };

		pos.null_move(ep_copy, capt_copy);
		nodes += 1;

		no_pruning[depth + 1] = true;
		score = -alphabeta(pos, ply - R - 1, depth + 1, 1 - beta, -beta);
		no_pruning[depth + 1] = false;

		if (engine::stop)
			return no_score;

		pos.undo_null_move(ep_copy, capt_copy);

		if (score >= beta)
		{
			hash::store(pos, no_move, score, ply, depth, lower);
			return beta;
		}
	}

	int fut_eval{ no_score };
	if (ply <= 6 && !pv_node && !in_check)
	{
		assert(score != NO_SCORE);
		fut_eval = score + 50 + 100 * ply;
	}

	movepick pick(pos, tt_move, history, killer, depth);

	const int prev_alpha{ alpha };
	int move_nr{ 0 };

	for (auto move{ pick.next() }; move != nullptr; move_nr += 1, move = pick.next())
	{
		assert(pick.list.is_pseudolegal(*move));

		pos.new_move(*move);

		if (!is_legal(pos))
		{
			assert(pick.list.mode == PSEUDO);
			pos = pick.reverse;
			move_nr -= 1;
			continue;
		}

		nodes += 1;

		int ext{ 0 };
		const bool new_check{ is_check(pos) };

		if (new_check && (ply <= 4 || pv_node))
			ext = 1;
		else if (pv_node && pick.reverse.recapture(*move))
			ext = 1;
		else if (pv_node && is_pawn_to_7th(*move))
			ext = 1;

		if (move_nr > 0 && fut_eval <= alpha && is_quiet(*move) && !is_mate(alpha) && !new_check && !is_killer(
			*move, depth))
		{
			pos = pick.reverse;
			continue;
		}

		if (pv_node && move_nr > 0)
		{
			score = -alphabeta(pos, ply - 1 + ext, depth + 1, -alpha, -alpha - 1);

			if (score > alpha)
				score = -alphabeta(pos, ply - 1 + ext, depth + 1, -alpha, -beta);
		}
		else
			score = -alphabeta(pos, ply - 1 + ext, depth + 1, -alpha, -beta);

		if (engine::stop)
			return no_score;

		pos = pick.reverse;

		if (score >= beta)
		{
			hash::store(pos, no_move, score, ply, depth, lower);
			heuristics(pos, *move, ply, depth);

			return beta;
		}

		if (score > alpha)
		{
			alpha = score;

			hash::store(pos, *move, score, ply, depth, exact);
			temp_pv(depth, *move);
		}
	}

	if (alpha == prev_alpha)
	{
		if (move_nr == 0)
			return in_check ? depth - max_score : contempt[pos.turn];
		hash::store(pos, no_move, alpha, ply, depth, upper);
	}

	return alpha;
}

int search::root_alphabeta(board& pos, movepick& pick, uint32_t pv[], const int ply)
{
	assert(ply <= lim::depth);

	if (pv[0] != no_move)
	{
		assert(pick.list.in_list(pv[0]));
		pick.rearrange_root(root_nodes, pick.list.find(pv[0]));

		pv[0] = no_move;
	}

	constexpr int beta{ max_score };
	int alpha{ -beta };

	int score;
	int move_nr{ 0 };

	for (auto move{ pick.next() }; move != nullptr; move_nr += 1, move = pick.next())
	{
		assert(pick.list.is_pseudolegal(*move));
		assert(beta > alpha);
		assert(pick.list.in_list(*move));

		currmove(ply, *move, move_nr + 1);

		const auto real_nr{ static_cast<uint32_t>(move - pick.list.movelist) };

		root_nodes[real_nr] -= nodes;
		pos.new_move(*move);
		nodes += 1;

		assert(is_legal(pos));

		if (move_nr > 0)
		{
			score = -alphabeta(pos, ply - 1, 1, -alpha, -alpha - 1);

			if (score > alpha)
				score = -alphabeta(pos, ply - 1, 1, -alpha, -beta);
		}
		else
			score = -alphabeta(pos, ply - 1, 1, -alpha, -beta);

		root_nodes[real_nr] += nodes;
		pos = pick.reverse;

		if (engine::stop)
			break;

		if (score > alpha)
		{
			alpha = score;

			main_pv(*move, pv, static_cast<int>(real_nr));
			hash::store(pos, pv[0], score, ply, 0, exact);
		}
	}

	return alpha;
}

int search::qsearch(board& pos, int alpha, const int beta)
{
	if (by_material(pos))
		return contempt[pos.turn];

	if (stop_thread())
	{
		engine::stop = true;
		return no_score;
	}

	int score{ eval::static_eval(pos) };

	if (score >= beta)
		return beta;

	if (score > alpha)
		alpha = score;

	movepick pick(pos);

	for (auto move{ pick.next() }; move != nullptr; move = pick.next())
	{
		assert(pick.list.is_pseudolegal(*move));

		if (!pos.lone_king()
			&& !is_promo(*move)
			&& score + see::exact_value[to_victim(*move)] + 100 < alpha)
			continue;

		if (!is_promo(*move)
			&& see::exact_value[to_piece(*move)] > see::exact_value[to_victim(*move)]
			&& see::eval(pos, *move) < 0)
			continue;

		pos.new_move(*move);
		nodes += 1;

		assert(is_legal(pos));

		score = -qsearch(pos, -beta, -alpha);

		pos = pick.reverse;

		if (engine::stop)
			return no_score;

		if (score >= beta)
			return beta;

		if (score > alpha)
			alpha = score;
	}

	return alpha;
}

uint32_t search::id_frame(board& pos, timemanager& chrono, uint32_t& ponder)
{
	assert(engine::depth >= 1);
	assert(engine::depth <= lim::depth);

	mean_time.call_cnt = 0;
	mean_time.max = chrono.get_movetime(pos.turn);
	mean_time.manage.start();

	uint32_t pv[lim::depth]{ 0 };
	uint32_t best_move{ 0 };

	contempt[pos.turn] = -engine::contempt;
	contempt[pos.not_turn] = engine::contempt;

	for (int i{ 0 }; i < engine::move_cnt; ++i)
		hashlist[i] = engine::hashlist[i];

	perft.prev = 0;
	perft.all = 0;

	for (auto& i : pv_evol) for (auto& p : i) p = no_move;
	for (auto& i : killer.list) for (auto& k : i) k = no_move;
	for (auto& i : history.list) for (auto& j : i) for (auto& h : j) h = 1000ULL;

	movepick pick(pos, root_nodes);

	nodes = 0;
	if (pick.list.cnt.moves == 1)
		return pick.list.movelist[0];

	for (int ply{ 1 }; ply <= engine::depth; ++ply)
	{
		int score{ root_alphabeta(pos, pick, pv, ply) };

		auto interim{ mean_time.manage.elapsed() };

		if (pv[0])
		{
			assert(nodes != 0);
			auto nps{ nodes * 1000 / (interim + 1) };

			int mindepth{ engine::stop ? ply - 1 : ply };
			int maxdepth{ 0 };
			int seldepth{ ply };

			while (seldepth < lim::depth - 1 && pv[seldepth] != no_move)
				seldepth += 1;

			for (int d{ max_score - abs(score) }; d <= seldepth - 1; pv[d++] = no_move);

			for (; maxdepth < lim::depth && pv[maxdepth] != no_move; maxdepth += 1)
			{
				if (!pick.list.is_pseudolegal(pv[maxdepth]))
				{
					for (int d{ maxdepth++ }; d < lim::depth && pv[d] != no_move; pv[d++] = no_move);
					break;
				}

				pos.new_move(pv[maxdepth]);

				if (!is_legal(pos))
				{
					for (int d{ maxdepth++ }; d < lim::depth && pv[d] != no_move; pv[d++] = no_move);
					break;
				}
			}
			pos = pick.reverse;

			best_move = pv[0];
			ponder = pv[1];

			for (int d{ 0 }, neg{ 1 }; d < lim::depth && pv[d] != no_move; ++d, neg *= -1)
			{
				assert(pick.list.is_pseudolegal(pv[d]));
				assert(maxdepth - d > 0);

				hash::store(pos, pv[d], score * neg, maxdepth - d + 1, d, exact);

				pos.new_move(pv[d]);

				assert(is_legal(pos));
			}
			pos = pick.reverse;

			if (perft.prev != 0)
			{
				perft.factor += static_cast<double>(nodes - perft.all) / static_cast<double>(perft.prev);
				perft.cnt += 1;
			}
			perft.prev = nodes - perft.all;
			perft.all = nodes;
			perft.total += perft.prev;

			std::string score_str{"cp " + std::to_string(score)};
			if (is_mate(score))
				score_str = "mate " + std::to_string((max_score - abs(score) + 1) / 2);

			std::cout << "info"
				<< " depth " << mindepth
				<< " seldepth " << seldepth
				<< " score " << score_str
				<< " time " << interim
				<< " nodes " << nodes
				<< " nps " << nps
				<< " hashfull " << hash::hashfull()
				<< " pv ";
			for (int d{ 0 }; d < ply && pv[d] != no_move; ++d)
				std::cout << algebraic(pv[d]) << " ";

			std::cout << std::endl;

			if (engine::stop)
				break;
			if (engine::infinite)
				continue;
			if (score > mate_score)
				break;
		}

		else break;

		if (ply > 8 && pv[ply - 8] == no_move)
			break;
		if (interim > mean_time.max / 2)
			break;
	}

	return best_move;
}

bool search::is_legal(const board& pos)
{
	return attack::check(pos, pos.not_turn, pos.pieces[King] & pos.side[pos.not_turn]) != 0ULL;
}

void search::stop_ponder()
{
	mean_time.max += mean_time.manage.elapsed();
}