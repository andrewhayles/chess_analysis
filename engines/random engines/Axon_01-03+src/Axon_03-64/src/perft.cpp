
#include "attack.h"
#include "board.h"
#include "common.h"
#include "movegen.h"
#include "search.h"
#include "perft.h"

namespace perft
{
	void root_perft(board& pos, const int depth, const gen_mode mode)
	{
		assert(depth >= 1 && depth <= lim::depth);
		assert(mode == LEGAL || mode == PSEUDO);

		search::mean_time.manage.start();
		search::perft.total += search::perft.all;
		search::perft.all = 0;

		for (int d{ 1 }; d <= depth; ++d)
		{
			std::cout << "perft " << d << ": ";

			const uint64_t nodes{ perft(pos, d, mode) };
			const auto interim{search::mean_time.manage.elapsed() };

			search::perft.all += nodes;

			std::cout.precision(3);
			std::cout << nodes
				<< " time " << interim
				<< " nps " << std::fixed << static_cast<double>(search::perft.all)
					/ (static_cast<double>(interim + 1)) / 1000.0 << "M" << std::endl;
		}
	}

	uint64_t perft(board& pos, const int depth, const gen_mode mode)
	{
		uint64_t nodes{ 0 };

		if (depth == 0) return 1;

		movegen list(pos, mode);
		list.gen_tactical(), list.gen_quiet();

		const board saved(pos);

		for (int i{ 0 }; i < list.cnt.moves; ++i)
		{
			assert(list.is_pseudolegal(list.movelist[i]));

			pos.new_move(list.movelist[i]);

			if (mode == legal)
				assert(is_legal(pos));
			if (mode == pseudo && !search::is_legal(pos))
			{
				pos = saved;
				continue;
			}

			nodes += perft(pos, depth - 1, mode);
			pos = saved;
		}

		return nodes;
	}

	void reset()
	{
		search::perft.prev = 0;
		search::perft.all = 0;
		search::perft.total = 0;
		search::perft.factor = 0;
		search::perft.cnt = 0;
	}

	void summary()
	{
		search::perft.factor /= (search::perft.cnt != 0 ? search::perft.cnt : 1);
	}
}
