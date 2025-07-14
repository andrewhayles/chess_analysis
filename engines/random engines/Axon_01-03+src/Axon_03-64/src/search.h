#pragma once
#include "attack.h"
#include "bitops.h"
#include "board.h"
#include "common.h"
#include "engine.h"
#include "movepick.h"
#include "timemanager.h"

namespace search
{
	int alphabeta(board& pos, int ply, int depth, int beta, int alpha);
	int qsearch(board& pos, int alpha, int beta);
	int root_alphabeta(board& pos, movepick& pick, uint32_t pv[], int ply);
	bool is_legal(const board& pos);
	uint32_t id_frame(board& pos, timemanager& chrono, uint32_t& ponder);
	void stop_ponder();
	inline uint32_t pv_evol[lim::depth][lim::depth];
	inline uint64_t hashlist[lim::period];
	inline bool no_pruning[lim::depth]{ false };
	inline int contempt[]{ 0, 0 };
	inline sort::killer_list killer{};
	inline sort::history_list history{};
	inline constexpr uint64_t max_history{ 1ULL << 63 };
	inline uint64_t root_nodes[lim::movegen]{ 0 };
	inline uint64_t nodes{ 0 };

	inline struct perft_result
	{
		uint64_t all;
		uint64_t total;
		uint64_t prev;
		double factor;
		int cnt;
		perft_result() = default;
	} perft;

	inline struct time_management
	{
		int call_cnt{ 0 };
		uint64_t max{};
		timer manage;
		time_management() = default;
	} mean_time;

	inline bool stop_thread()
	{
		if (++search::mean_time.call_cnt < 256)
			return false;

		search::mean_time.call_cnt = 0;
		if (engine::infinite) return false;

		if (search::nodes >= engine::nodes) return true;

		return search::mean_time.manage.elapsed() >= search::mean_time.max;
	}

	inline bool is_check(const board& pos)
	{
		return attack::check(pos, pos.turn, pos.pieces[King] & pos.side[pos.turn]) == 0ULL;
	}

	inline bool is_mate(const int score)
	{
		return abs(score) > mate_score;
	}

	inline bool is_promo(const uint32_t move)
	{
		static_assert(promo_queen == 15, "promo encoding");
		static_assert(promo_knight == 12, "promo encoding");

		return to_flag(move) >= 12;
	}

	inline bool is_quiet(const uint32_t move)
	{
		return to_victim(move) == no_piece && !is_promo(move);
	}

	inline bool is_killer(const uint32_t move, const int depth)
	{
		return move == search::killer.list[depth][0] || move == search::killer.list[depth][1];
	}

	inline bool is_pawn_to_7th(const uint32_t move)
	{
		return to_piece(move) == Pawn && ((1ULL << to_sq2(move)) & (rank[r2] | rank[r7]));
	}

	inline void currmove(const int ply, const uint32_t move, const int move_nr)
	{
		if (const auto interim{ search::mean_time.manage.elapsed() }; (interim > 1000 && move_nr <= 3) || interim > 5000)
		{
			std::cout << "info depth " << ply
				<< " currmove " << algebraic(move)
				<< " currmovenumber " << move_nr
				<< " time " << interim
				<< " nps " << search::nodes * 1000 / interim
				<< std::endl;
		}
	}

	inline void killer_moves(const uint32_t move, const int depth)
	{
		if (move == search::killer.list[depth][0] || move == search::killer.list[depth][1])
			return;

		search::killer.list[depth][1] = search::killer.list[depth][0];
		search::killer.list[depth][0] = move;
	}

	inline void history_table(const board& pos, const uint32_t move, const int ply)
	{
		assert(pos.turn == to_turn(move));

		uint64_t* entry{ &(search::history.list[pos.turn][to_piece(move)][to_sq2(move)]) };
		*entry += static_cast<unsigned long long>(ply) * ply;

		if (*entry > search::max_history)
		{
			for (auto& i : search::history.list) for (auto& j : i) for (auto& h : j) h >>= 1;
		}
	}

	inline void heuristics(const board& pos, const uint32_t move, const int ply, const int depth)
	{
		if (is_quiet(move))
		{
			history_table(pos, move, ply);
			killer_moves(move, depth);
		}
	}

	inline void temp_pv(const int depth, const uint32_t move)
	{
		search::pv_evol[depth - 1][0] = move;
		for (int i{ 0 }; search::pv_evol[depth][i] != no_move; ++i)
		{
			search::pv_evol[depth - 1][i + 1] = search::pv_evol[depth][i];
		}
	}

	inline void main_pv(const uint32_t move, uint32_t pv[], const int real_nr)
	{
		pv[0] = move;
		for (int i{ 0 }; search::pv_evol[0][i] != no_move; ++i)
		{
			pv[i + 1] = search::pv_evol[0][i];
		}

		search::root_nodes[real_nr] += search::nodes;
	}
}

constexpr uint64_t all_sq[]{ 0xaa55aa55aa55aa55, 0x55aa55aa55aa55aa };

inline bool lone_bishops(const board& pos)
{
	return (pos.pieces[Bishop] | pos.pieces[King]) == pos.side[both];
}

inline bool lone_knights(const board& pos)
{
	return (pos.pieces[Knight] | pos.pieces[King]) == pos.side[both];
}

inline bool by_repetition(const board& pos, const int depth)
{
	const int size{ engine::move_cnt + depth - 1 };
	for (int i{ 4 }; i <= pos.half_move_cnt && i <= size; i += 2)
	{
		if (search::hashlist[size - i] == search::hashlist[size])
			return true;
	}

	return false;
}

inline bool by_material(const board& pos)
{
	if (lone_bishops(pos) && (!(all_sq[white] & pos.pieces[Bishop]) || !(all_sq[black] & pos.pieces[
		Bishop])))
		return true;

		if (lone_knights(pos) && popcnt(pos.pieces[Knight]) == 1)
			return true;

		return false;
}

inline bool verify(const board& pos, const int depth)
{
	search::hashlist[pos.move_cnt - 1] = pos.key;

	if (pos.half_move_cnt >= 4 && by_repetition(pos, depth))
		return true;
	if (by_material(pos))
		return true;
	if (pos.half_move_cnt == 100)
		return true;
	return false;
}
