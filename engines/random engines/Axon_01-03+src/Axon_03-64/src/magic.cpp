#include "magic.h"

#include "bitops.h"
#include "random.h"

void magic::init()
{
	attack_table.clear();
	attack_table.resize(table_size);

	std::vector<uint64_t> attack_temp;
	attack_temp.reserve(table_size);

	std::vector<uint64_t> blocker;
	blocker.reserve(table_size);

	for (int sl{ ROOK }; sl <= BISHOP; ++sl)
	{
		init_mask(sl);
		init_blocker(sl, blocker);
		init_move(sl, blocker, attack_temp);
		init_magic(sl, blocker, attack_temp);
		connect(sl, blocker, attack_temp);
	}
}

static void real_shift(uint64_t& bb, const int shift)
	{
		bb = (bb << shift) | (bb >> (64 - shift));
	}

void magic::init_mask(const int sl)
{
	assert(sl == ROOK || sl == BISHOP);

	for (int sq{ 0 }; sq < 64; ++sq)
	{
		const uint64_t sq64{ 1ULL << sq };

		for (int dir{ sl }; dir < 8; dir += 2)
		{
			uint64_t flood{ sq64 };
			while (!(flood & ray[dir].border))
			{
				slider[sl][sq].mask |= flood;
				real_shift(flood, ray[dir].shift);
			}
		}
		slider[sl][sq].mask ^= sq64;
	}
}

void magic::init_blocker(const int sl, std::vector<uint64_t>& blocker)
{
	assert(sl == ROOK || sl == BISHOP);
	assert(blocker.size() == 0 || blocker.size() == 102400);

	bool bit[12]{ false };

	for (int sq{ 0 }; sq < 64; ++sq)
	{
		slider[sl][sq].offset = blocker.size();

		uint64_t mask_split[12]{ 0 };
		int bits_in{ 0 };

		uint64_t mask_bit{ slider[sl][sq].mask };
		while (mask_bit)
		{
			mask_split[bits_in++] = 1ULL << lsb(mask_bit);

			mask_bit &= mask_bit - 1;
		}
		assert(bits_in <= 12);
		assert(bits_in >= 5);
		assert(popcnt(slider[sl][sq].mask) == bits_in);

		slider[sl][sq].shift = 64 - bits_in;

		const int max{ 1 << bits_in };
		for (int a{ 0 }; a < max; ++a)
		{
			uint64_t pos{ 0 };
			for (int b{ 0 }; b < bits_in; ++b)
			{
				if (!(a % (1 << b)))
					bit[b] = !bit[b];
				if (bit[b])
					pos |= mask_split[b];
			}
			blocker.push_back(pos);
		}
	}
}

void magic::connect(const int sl, const std::vector<uint64_t>& blocker, const std::vector<uint64_t>& attack_temp)
{
	assert(sl == ROOK || sl == BISHOP);

	for (int sq{ 0 }; sq < 64; ++sq)
	{
		const int max{ 1 << (64 - slider[sl][sq].shift) };

		for (int cnt{ 0 }; cnt < max; ++cnt)
		{
			attack_table
				[
					static_cast<uint32_t>(slider[sl][sq].offset
					+ (blocker[slider[sl][sq].offset + cnt] * slider[sl][sq].magic >> slider[sl][sq].shift))
				]
			= attack_temp[slider[sl][sq].offset + cnt];
		}
	}
}
void magic::init_move(const int sl, const std::vector<uint64_t>& blocker, std::vector<uint64_t>& attack_temp)
{
	assert(sl == ROOK || sl == BISHOP);
	assert(attack_temp.size() == 0 || attack_temp.size() == 102400);

	for (int sq{ 0 }; sq < 64; ++sq)
	{
		const uint64_t sq64{ 1ULL << sq };

		const int max{ 1 << (64 - slider[sl][sq].shift) };
		for (int cnt{ 0 }; cnt < max; ++cnt)
		{
			uint64_t pos{ 0 };

			for (int dir{ sl }; dir < 8; dir += 2)
			{
				uint64_t flood{ sq64 };
				while (!(flood & ray[dir].border) && !(flood & blocker[slider[sl][sq].offset + cnt]))
				{
					real_shift(flood, ray[dir].shift);
					pos |= flood;
				}
			}
			attack_temp.push_back(pos);

			assert(attack_temp.size() - 1 == slider[sl][sq].offset + cnt);
		}
	}
}

void magic::init_magic(const int sl, const std::vector<uint64_t>& blocker, const std::vector<uint64_t>& attack_temp)
{
	constexpr uint64_t seeds[]
	{
		908859, 953436, 912753, 482262, 322368, 711868, 839234, 305746,
		711822, 703023, 270076, 964393, 704635, 626514, 970187, 398854
	};

	bool fail;
	for (int sq{ 0 }; sq < 64; ++sq)
	{
		const int occ_size{ 1 << (64 - slider[sl][sq].shift) };
		assert(occ_size <= 4096);

		std::vector<uint64_t> occ;
		occ.resize(occ_size);

		rand_xor rand_gen{ seeds[sq >> 2] };

		do
		{
			do
			{
				slider[sl][sq].magic = rand_gen.sparse64();
			} while (popcnt((slider[sl][sq].mask * slider[sl][sq].magic) & 0xff00000000000000) < 6);

			fail = false;
			occ.clear();
			occ.resize(occ_size);

			for (int i{ 0 }; !fail && i < occ_size; ++i)
			{
				const int idx{
					static_cast<int>(blocker[slider[sl][sq].offset + i] * slider[sl][sq].magic >> slider[sl][sq].shift)
				};
				assert(idx <= occ_size);

				if (!occ[idx])
					occ[idx] = attack_temp[slider[sl][sq].offset + i];

				else if (occ[idx] != attack_temp[slider[sl][sq].offset + i])
				{
					fail = true;
					break;
				}
			}
		} while (fail);
	}
}

