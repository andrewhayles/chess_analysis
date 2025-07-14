#include "hash.h"

#include "bitops.h"
#include "engine.h"

void hash::clear()
{
	for (uint32_t i{ 0 }; i < tt_size; ++i)
		table_[i] = { 0ULL, no_score, no_move, 0, 0, 0, 0 };
}

void hash::erase()
{
	if (table_ != nullptr)
	{
		delete[] table_;
		table_ = nullptr;
	}
}

int hash::create(uint64_t size)
{
	erase();
	if (size > lim::hash)
		size = lim::hash;
	auto size_temp{ ((size << 20) / sizeof(trans)) >> 1 };

	tt_size = 1ULL;
	for (; tt_size <= size_temp; tt_size <<= 1);

	zobrist::mask = tt_size - zobrist::slots;

	table_ = new trans[tt_size];
	clear();
	size_temp = (tt_size * sizeof(trans)) >> 20;

	assert(size_temp <= size && size_temp <= lim::hash);
	return static_cast<int>(size_temp);
}

int hash::hashfull()
{
	if (tt_size < 1000) return 0;
	int per_mill{ 0 };

	for (int i{ 0 }; i < 1000; ++i)
	{
		if (table_[i].key != 0)
			++per_mill;
	}

	return per_mill;
}

bool hash::probe(const board& pos, uint32_t& move, int& score, const int ply, const int depth, uint8_t& flag)
{
	assert(score == NO_SCORE);

	trans* entry{ &table_[pos.key & zobrist::mask] };

	for (int i{ 0 }; i < zobrist::slots; ++i, ++entry)
	{
		if (entry->key == pos.key)
		{
			assert(entry->ply <= lim::depth);

			entry->age = static_cast<uint8_t>(engine::move_cnt);
			move = entry->move | (entry->info << 16);
			flag = entry->flag;

			if (entry->ply >= ply)
			{
				score = entry->score;

				if (score > mate_score) score -= depth;
				if (score < -mate_score) score += depth;

				assert(abs(score) <= MAX_SCORE);
				return true;
			}
			return false;
		}
	}

	return false;
}

void hash::store(const board& pos, const uint32_t move, int score, const int ply, const int depth,
	const uint8_t flag)
{
	if (abs(score) == abs(engine::contempt)) return;

	if (score > mate_score) score += depth;
	if (score < -mate_score) score -= depth;

	assert(abs(score) <= MAX_SCORE);
	assert(ply <= lim::depth);
	assert(ply >= 0);

	trans* entry{ &table_[pos.key & zobrist::mask] };
	trans* replace{ entry };

	auto lowest{ static_cast<uint32_t>(lim::depth) + (engine::move_cnt << 8) };

	for (int i{ 0 }; i < zobrist::slots; ++i, ++entry)
	{
		if (entry->key == pos.key)
		{
			entry->score = static_cast<int16_t>(score);
			entry->move = static_cast<uint16_t>(move);
			entry->info = static_cast<uint8_t>(move >> 16);
			entry->age = static_cast<uint8_t>(engine::move_cnt);
			entry->ply = static_cast<uint8_t>(ply);
			entry->flag = flag;
			return;
		}

		if (entry->key == 0ULL)
		{
			replace = entry;
			break;
		}

		const auto new_low{ entry->ply + ((entry->age + (abs(engine::move_cnt - entry->age) & ~0xffU)) << 8) };

		assert(entry->ply <= lim::depth);

		if (new_low < lowest)
		{
			lowest = new_low;
			replace = entry;
		}
	}

	replace->key = pos.key;
	replace->score = static_cast<int16_t>(score);
	replace->move = static_cast<uint16_t>(move);
	replace->info = static_cast<uint8_t>(move >> 16);
	replace->age = static_cast<uint8_t>(engine::move_cnt);
	replace->ply = static_cast<uint8_t>(ply);
	replace->flag = flag;
}

void hash::reset()
{
	erase();
	table_ = new trans[tt_size];
	clear();
}

uint64_t zobrist::to_key(const board& pos)
{
	uint64_t key{ 0 };

	for (int col{ white }; col <= black; ++col)
	{
		uint64_t pieces{ pos.side[col] };
		while (pieces)
		{
			const auto sq_old{ lsb(pieces) };
			const auto sq_new{ mirror(sq_old) };
			const auto piece{ (pos.piece_sq[sq_old] << 1) + (col ^ 1) };

			assert(pos.piece_sq[sq_old] != NONE);

			key ^= rand_key[(piece << 6) + sq_new];
			pieces &= pieces - 1;
		}
	}

	for (int i{ 0 }; i < 4; ++i)
	{
		if (pos.castl_rights[i])
			key ^= rand_key[offset.castling + order[i]];
	}

	if (pos.ep_sq)
	{
		if (const auto file_idx{ lsb(pos.ep_sq) & 7 }; pos.pieces[Pawn] & pos.side[pos.turn] & ep_flank[pos
			.
			not_turn][file_idx])
			key ^= rand_key[offset.ep + 7 - file_idx];
	}

	key ^= is_turn[pos.turn];

	return key;
}

