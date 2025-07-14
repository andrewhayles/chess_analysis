#pragma once
#include "board.h"

namespace eval
{
	constexpr int bishop_mob[2][14]
	{
		{-15, -10, 0, 10, 18, 25, 30, 34, 37, 39, 40, 41, 42, 43},
		{-15, -10, 0, 10, 18, 25, 30, 34, 37, 39, 40, 41, 42, 43}
	};

	constexpr int bishop_pair[2]
	{ 15, 30 };

	constexpr int king_safety_w[8]
	{
		0, 0, 50, 75, 88, 94, 97, 99
	};

	constexpr int king_threat[5]
	{
		0, 20, 20, 40, 80
	};

	constexpr int knight_mob[2][9]
	{
		{-20, -15, -8, 0, 8, 12, 15, 17, 18},
		{-20, -15, -8, 0, 8, 12, 15, 17, 18}
	};

	constexpr int knights_connected
	{ 10 };

	constexpr int major_behind_pp[2]
	{ 10, 20 };

	static int passed_pawn[2][8]
	{
		{0, 0, 0, 20, 60, 110, 170, 0}
	};

	constexpr int queen_mob[2][28]
	{
		{
			-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 6, 7, 7,
			8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8
		},
		{
			-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 13, 14,
			14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14
		}
	};

	constexpr int rook_mob[2][15]
	{
		{-8, -5, -3, 0, 3, 5, 8, 10, 12, 13, 14, 15, 16, 17, 18},
		{-18, -10, -4, 2, 8, 14, 20, 26, 32, 37, 40, 42, 43, 44, 45}
	};

	constexpr int rook_on_7th[2]
	{ 10, 20 };

	constexpr int rook_open_file
	{ 25 };

	constexpr int pc_value[2][6]
	{
		{90, 320, 330, 500, 1000, 0},
		{100, 320, 350, 550, 1050, 0}
	};

	void init();
	void pieces(const board& pos, int sum[][2]);
	void pawns(const board& pos, int sum[][2]);
	int static_eval(const board& pos);

	constexpr int max_weight{ 42 };
	constexpr int negate[]{ 1, -1 };
	constexpr uint32_t seventh_rank[]{ r7, r2 };
	inline uint64_t front_span[2][64]{};

	static_assert(white == 0, "index");
	static_assert(black == 1, "index");
	static_assert(mg == 0, "index");
	static_assert(eg == 1, "index");

	inline struct pinned
	{
		int cnt;
		uint32_t idx[8];
		uint64_t moves[64];
	} pin;
}