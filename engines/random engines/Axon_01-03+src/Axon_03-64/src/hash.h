#pragma once
#include "board.h"
#include "rand_key.h"

class hash
{
	struct trans
	{
		uint64_t key;
		int16_t score;
		uint16_t move;
		uint8_t info;
		uint8_t age;
		uint8_t ply;
		uint8_t flag;
	};

	static trans* table_;

	static_assert(sizeof(trans) == 16, "tt entry size of 16 bytes");

public:
	explicit hash(const uint64_t size)
	{
		erase();
		create(size);
	}

	~hash() { erase(); }
	static uint64_t tt_size;

private:
	static void clear();
	static void erase();

public:
	static int create(uint64_t size);
	static int hashfull();
	static bool probe(const board& pos, uint32_t& move, int& score, int ply, int depth, uint8_t& flag);
	static void store(const board& pos, uint32_t move, int score, int ply, int depth, uint8_t flag);
	static void reset();
};

inline hash::trans* hash::table_{nullptr};
inline uint64_t hash::tt_size{0};

namespace zobrist
{
	constexpr uint64_t ep_flank[][8]
	{
		{
			0x0002000000, 0x0005000000,
			0x000a000000, 0x0014000000,
			0x0028000000, 0x0050000000,
			0x00a0000000, 0x0040000000
		},
		{
			0x0200000000, 0x0500000000,
			0x0a00000000, 0x1400000000,
			0x2800000000, 0x5000000000,
			0xa000000000, 0x4000000000
		}
	};

	constexpr struct key_offset
	{
		int castling{ 768 };
		int ep{ 772 };
		int turn{ 780 };
	} offset;

	constexpr uint64_t is_turn[]
	{
		rand_key[offset.turn],
		0ULL
	};

	inline uint32_t mirror(const uint32_t sq)
	{
		return (sq & 56) - (sq & 7) + 7;
	}

	uint64_t to_key(const board& pos);
	constexpr int order[]{ 0, 2, 1, 3 };
	constexpr int slots{ 4 };
	inline uint64_t mask;
}