#pragma once
#include "board.h"
#include "timemanager.h"

class engine
{
public:
	static bool infinite;
	static bool stop;
	static const std::string startpos;
	static int contempt;
	static int depth;
	static int hash_size;
	static int move_cnt;
	static uint32_t movelist[];
	static uint32_t start_searching(board& pos, timemanager& chrono, uint32_t& ponder);
	static uint64_t hashlist[];
	static uint64_t nodes;
	static void clear_hash();
	static void init_eval();
	static void init_magic();
	static void init_movegen();
	static void new_game(board& pos);
	static void new_hash_size(int size);
	static void new_move(board& pos, uint32_t move);
	static void parse_fen(board& pos, const std::string& fen);
	static void reset_game();
	static void save_move(const board& pos, uint32_t move);
	static void stop_ponder();
};

inline const std::string engine::startpos
{
	"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
};

inline bool engine::infinite{false};
inline bool engine::stop{true};
inline int engine::contempt{0};
inline int engine::depth;
inline int engine::hash_size{128};
inline int engine::move_cnt;
inline uint32_t engine::movelist[lim::period];
inline uint64_t engine::hashlist[lim::period];
inline uint64_t engine::nodes;