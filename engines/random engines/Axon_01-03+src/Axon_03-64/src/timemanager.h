#pragma once
#include <chrono>

class timemanager
{
public:
	timemanager() : incr{0}, moves_to_go{50}, time{0}
	{
	}

	int incr[2];
	int moves_to_go;
	int time[2];
	uint64_t get_movetime(int turn);
	uint64_t movetime{};
	void set_movetime(uint64_t new_time);
};

class timer
{
	std::chrono::time_point<std::chrono::system_clock> start_time_;

public:
	timer() { start(); }
	void start();
	[[nodiscard]] uint64_t elapsed() const;
};
