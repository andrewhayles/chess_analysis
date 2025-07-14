#include "eval.h"

#include "attack.h"
#include "bitops.h"
#include "magic.h"
#include "movegen.h"
#include "pst.h"

namespace
{
	void pin_down(const board& pos, const int turn)
	{
		eval::pin.cnt = 0;
		const int king_sq{ pos.king_sq[turn] };
		const int not_turn{ turn ^ 1 };
		const uint64_t fr_king = pos.pieces[King] & pos.side[turn];

		uint64_t all_att{
			movegen::slide_ray[ROOK][king_sq] & pos.side[not_turn] & (pos.pieces[Rook] | pos.pieces[Queen])
		};
		all_att |= movegen::slide_ray[BISHOP][king_sq] & pos.side[not_turn] & (pos.pieces[Bishop] | pos.pieces[
			Queen]);

		while (all_att)
		{
			uint64_t ray_to_att{ attack::by_slider(ROOK, king_sq, all_att) };
			ray_to_att |= attack::by_slider(BISHOP, king_sq, all_att);

			const uint64_t att{ 1ULL << lsb(all_att) };

			if (!(att & ray_to_att))
			{
				all_att &= all_att - 1;
				continue;
			}

			assert(fr_king);

			uint64_t x_ray{ 0 };
			for (int dir{ 0 }; dir < 8; ++dir)
			{
				auto flood{ fr_king };
				for (; !(flood & magic::ray[dir].border); flood |= shift(flood, magic::ray[dir].shift));

				if (flood & att)
				{
					x_ray = flood & ray_to_att;
					break;
				}
			}

			assert(x_ray & att);
			assert(!(x_ray & fr_king));

			if (x_ray & pos.side[turn] && popcnt(x_ray & pos.side[both]) == 2)
			{
				assert(popcnt(x_ray & pos.side[turn]) == 1);

				const auto sq{ lsb(x_ray & pos.side[turn]) };
				eval::pin.moves[sq] = x_ray;
				eval::pin.idx[eval::pin.cnt++] = sq;
			}

			all_att &= all_att - 1;
		}
	}

	void unpin()
	{
		if (eval::pin.cnt != 0)
		{
			assert(pin.cnt <= 8);
			for (int i{ 0 }; i < eval::pin.cnt; ++i)
				eval::pin.moves[eval::pin.idx[i]] = ~0ULL;
		}
	}
}

void eval::init()
{
	std::fill_n(pin.moves, 64, ~0ULL);

	for (int p{ Pawn }; p <= King; ++p)
	{
		for (int s{ mg }; s <= eg; ++s)
		{
			for (int i{ 0 }; i < 64; ++i)
			{
				psqt[white][p][s][63 - i] += pc_value[s][p];
				psqt[black][p][s][i] = psqt[white][p][s][63 - i];
			}
		}
	}

	for (int i{ 0 }; i < 8; ++i)
	{
		passed_pawn[black][i] = passed_pawn[white][7 - i];
	}

	for (int i{ 8 }; i < 56; ++i)
	{
		uint64_t files{ file[i & 7] };
		if (i % 8)
			files |= file[(i - 1) & 7];
		if ((i - 7) % 8)
			files |= file[(i + 1) & 7];

		front_span[white][i] = files & ~((1ULL << (i + 2)) - 1);
		front_span[black][i] = files & ((1ULL << (i - 1)) - 1);
	}
}

int eval::static_eval(const board& pos)
{
	int sum[2][2]{};

	pieces(pos, sum);
	pawns(pos, sum);

	assert(pos.phase >= 0);
	const int weight{ pos.phase <= max_weight ? pos.phase : max_weight };
	const int mg_score{ sum[mg][white] - sum[mg][black] };
	const int eg_score{ sum[eg][white] - sum[eg][black] };

	const int fading{ pos.half_move_cnt <= 60 ? 40 : 100 - pos.half_move_cnt };

	return negate[pos.turn] * ((mg_score * weight + eg_score * (max_weight - weight)) / max_weight) * fading / 40;
}

void eval::pawns(const board& pos, int sum[][2])
{
	for (int col{ white }; col <= black; ++col)
	{
		const int not_col{ col ^ 1 };
		uint64_t pawns{ pos.pieces[Pawn] & pos.side[col] };
		while (pawns)
		{
			const auto sq{ lsb(pawns) };
			const auto idx{ sq & 7 };

			sum[mg][col] += psqt[not_col][Pawn][mg][sq];
			sum[eg][col] += psqt[not_col][Pawn][eg][sq];

			if (!(front_span[col][sq] & pos.pieces[Pawn] & pos.side[not_col]))
			{
				if (!(file[idx] & front_span[col][sq] & pos.pieces[Pawn]))
				{
					int mg_bonus{ passed_pawn[col][sq >> 3] };
					int eg_bonus{ mg_bonus };

					if (const uint64_t blocked_path{ file[idx] & front_span[col][sq] & pos.side[not_col] })
					{
						const int blocker_cnt{ popcnt(blocked_path) };
						mg_bonus /= blocker_cnt + 2;
						eg_bonus /= blocker_cnt + 1;
					}

					const uint64_t pieces_behind{ file[idx] & front_span[not_col][sq] };

					if (const uint64_t majors{ (pos.pieces[Rook] | pos.pieces[Queen]) & pos.side[col] }; (
						pieces_behind & majors) && !(pieces_behind & (pos.side[both] ^ majors)))
					{
						mg_bonus += major_behind_pp[mg];
						eg_bonus += major_behind_pp[eg];
					}

					sum[mg][col] += mg_bonus;
					sum[eg][col] += eg_bonus;
				}
			}
			pawns &= pawns - 1;
		}
	}
}

void eval::pieces(const board& pos, int sum[][2])
{
	for (int col{ white }; col <= black; ++col)
	{
		const int not_col{ col ^ 1 };
		int att_cnt{ 0 }, att_sum{ 0 };

		pin_down(pos, col);

		const uint64_t king_zone{ movegen::king_table[pos.king_sq[not_col]] };

		uint64_t pieces{ pos.side[col] & pos.pieces[Bishop] };
		while (pieces)
		{
			const auto sq{ lsb(pieces) };

			sum[mg][col] += psqt[not_col][Bishop][mg][sq];
			sum[eg][col] += psqt[not_col][Bishop][eg][sq];

			const uint64_t targets
			{
				attack::by_slider(BISHOP, sq, pos.side[both] & ~(pos.pieces[Queen] & pos.side[col]))
				& ~(pos.side[col] & pos.pieces[Pawn]) & pin.moves[sq]
			};

			if (const uint64_t streak_king{ targets & ~pos.side[both] & king_zone })
			{
				att_cnt += 1;
				att_sum += king_threat[Bishop] * popcnt(streak_king);
			}

			if (pieces & (pieces - 1))
			{
				sum[mg][col] += bishop_pair[mg];
				sum[eg][col] += bishop_pair[eg];
			}

			const int cnt{ popcnt(targets) };
			sum[mg][col] += bishop_mob[mg][cnt];
			sum[eg][col] += bishop_mob[eg][cnt];

			pieces &= pieces - 1;
		}

		pieces = pos.side[col] & pos.pieces[Rook];
		while (pieces)
		{
			const auto sq{ lsb(pieces) };

			sum[mg][col] += psqt[not_col][Rook][mg][sq];
			sum[eg][col] += psqt[not_col][Rook][eg][sq];

			const uint64_t targets
			{
				attack::by_slider(
					ROOK, sq, pos.side[both] & ~((pos.pieces[Queen] | pos.pieces[Rook]) & pos.side[col]))
				& ~(pos.side[col] & pos.pieces[Pawn]) & pin.moves[sq]
			};

			if (const uint64_t streak_king{ targets & ~pos.side[both] & king_zone })
			{
				att_cnt += 1;
				att_sum += king_threat[Rook] * popcnt(streak_king);
			}

			if (!(file[sq & 7] & pos.pieces[Pawn] & pos.side[col]))
			{
				sum[mg][col] += rook_open_file;

				if (!(file[sq & 7] & pos.pieces[Pawn]))
				{
					sum[mg][col] += rook_open_file;
				}
			}

			if (sq >> 3 == static_cast<int>(seventh_rank[col]))
			{
				if (rank[seventh_rank[col]] & pos.pieces[Pawn] & pos.side[not_col]
					|| rank[r8 * not_col] & pos.pieces[King] & pos.side[not_col])
				{
					sum[mg][col] += rook_on_7th[mg];
					sum[eg][col] += rook_on_7th[eg];
				}
			}

			const int cnt{ popcnt(targets) };
			sum[mg][col] += rook_mob[mg][cnt];
			sum[eg][col] += rook_mob[eg][cnt];

			pieces &= pieces - 1;
		}

		pieces = pos.side[col] & pos.pieces[Queen];
		while (pieces)
		{
			const auto sq{ lsb(pieces) };

			sum[mg][col] += psqt[not_col][Queen][mg][sq];
			sum[eg][col] += psqt[not_col][Queen][eg][sq];

			const uint64_t targets
			{
				(attack::by_slider(ROOK, sq, pos.side[both]) | attack::by_slider(BISHOP, sq, pos.side[both]))
				& ~(pos.side[col] & pos.pieces[Pawn]) & pin.moves[sq]
			};

			if (const uint64_t streak_king{ targets & ~pos.side[both] & king_zone })
			{
				att_cnt += 1;
				att_sum += king_threat[Queen] * popcnt(streak_king);
			}

			const int cnt{ popcnt(targets) };
			sum[mg][col] += queen_mob[mg][cnt];
			sum[eg][col] += queen_mob[eg][cnt];

			pieces &= pieces - 1;
		}

		const uint64_t pawn_att{ attack::by_pawns(pos, not_col) & ~pos.side[both] };
		pieces = pos.side[col] & pos.pieces[Knight];
		while (pieces)
		{
			const auto sq{ lsb(pieces) };

			sum[mg][col] += psqt[not_col][Knight][mg][sq];
			sum[eg][col] += psqt[not_col][Knight][eg][sq];

			const uint64_t targets
			{
				movegen::knight_table[sq]
				& ~(pos.side[col] & pos.pieces[Pawn]) & ~pawn_att & pin.moves[sq]
			};

			if (const uint64_t streak_king{ targets & ~pos.side[both] & king_zone })
			{
				att_cnt += 1;
				att_sum += king_threat[Knight] * popcnt(streak_king);
			}

			if (targets & pos.pieces[Knight] & pos.side[col])
			{
				sum[mg][col] += knights_connected;
				sum[eg][col] += knights_connected;
			}

			const int cnt{ popcnt(targets) };
			sum[mg][col] += knight_mob[mg][cnt];
			sum[eg][col] += knight_mob[eg][cnt];

			pieces &= pieces - 1;
		}

		sum[mg][col] += psqt[not_col][King][mg][pos.king_sq[col]];
		sum[eg][col] += psqt[not_col][King][eg][pos.king_sq[col]];

		const int score{ (king_safety_w[att_cnt & 7] * att_sum) / 100 };
		sum[mg][col] += score;
		sum[eg][col] += score;

		unpin();
	}
}
