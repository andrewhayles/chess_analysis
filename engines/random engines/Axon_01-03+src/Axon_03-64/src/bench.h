#pragma once
#include "common.h"

namespace bench
{
	struct unit
	{
		std::string fen;
		int max_depth;
	};

	static unit perft_pos[]
	{
		{"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -", 7},
	};

	void perft(std::string mode);
}