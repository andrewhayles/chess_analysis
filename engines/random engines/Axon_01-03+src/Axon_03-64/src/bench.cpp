#include "bench.h"

#include "board.h"
#include "engine.h"
#include "perft.h"

void bench::perft(std::string mode)
{
	board pos{};

	mode = (mode == "perft" ? "legal" : mode);
	const gen_mode gen_mode{ mode == "pseudolegal" ? pseudo : legal };

	perft::reset();

	for (auto& p : perft_pos)
	{
		std::cout << p.fen << std::endl;
		engine::parse_fen(pos, p.fen);

		perft::root_perft(pos, p.max_depth, gen_mode);
	}

	perft::summary();
}
