#pragma once
#include <vector>

#include "common.h"

class magic
{
	struct entry
	{
		uint64_t mask;
		uint64_t magic;
		size_t offset;
		int shift;
	};

public:
	struct pattern
	{
		int shift;
		uint64_t border;
	};

	static void init();
	static const int table_size;
	static const pattern ray[];
	static entry slider[2][64];
	static std::vector<uint64_t> attack_table;

private:
	static void connect(int sl, const std::vector<uint64_t>& blocker, const std::vector<uint64_t>& attack_temp);
	static void init_mask(int sl);
	static void init_blocker(int sl, std::vector<uint64_t>& blocker);
	static void init_move(int sl, const std::vector<uint64_t>& blocker, std::vector<uint64_t>& attack_temp);
	static void init_magic(int sl, const std::vector<uint64_t>& blocker, const std::vector<uint64_t>& attack_temp);
};

inline const magic::pattern magic::ray[]
{
	{8, 0xff00000000000000}, {7, 0xff01010101010101},
	{63, 0x0101010101010101}, {55, 0x01010101010101ff},
	{56, 0x00000000000000ff}, {57, 0x80808080808080ff},
	{1, 0x8080808080808080}, {9, 0xff80808080808080}
};

inline magic::entry magic::slider[2][64];
inline std::vector<uint64_t> magic::attack_table;
inline const int magic::table_size{107648};
