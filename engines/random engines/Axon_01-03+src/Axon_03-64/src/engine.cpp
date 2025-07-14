#include "engine.h"

#include "eval.h"
#include "hash.h"
#include "magic.h"
#include "movegen.h"
#include "perft.h"
#include "search.h"

void engine::clear_hash()
{
	hash::reset();
}

void engine::init_eval()
{
	eval::init();
}

void engine::init_magic()
{
	magic::init();
}

void engine::init_movegen()
{
	movegen::init_ray(ROOK);
	movegen::init_ray(BISHOP);
	movegen::init_king();
	movegen::init_knight();
}

void engine::new_game(board& pos)
{
	hash::reset();
	parse_fen(pos, startpos);
}

void engine::new_hash_size(const int size)
{
	hash_size = hash::create(size);
}

uint32_t engine::start_searching(board& pos, timemanager& chrono, uint32_t& ponder)
{
	perft::reset();
	return search::id_frame(pos, chrono, ponder);
}

void engine::new_move(board& pos, const uint32_t move)
{
	pos.new_move(move);
	save_move(pos, move);
}

void engine::parse_fen(board& pos, const std::string& fen)
{
	reset_game();
	pos.parse_fen(fen);
}

void engine::reset_game()
{
	move_cnt = 0;

	for (auto& m : movelist) m = no_move;
	for (auto& h : hashlist) h = 0ULL;
}

void engine::save_move(const board& pos, const uint32_t move)
{
	movelist[move_cnt] = move;
	hashlist[move_cnt] = pos.key;
	move_cnt += 1;
}

void engine::stop_ponder()
{
	infinite = false;
	search::stop_ponder();
}

