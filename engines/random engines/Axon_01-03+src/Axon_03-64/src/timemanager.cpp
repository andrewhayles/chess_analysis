#include "timemanager.h"

uint64_t timemanager::get_movetime(const int turn)
{
	if (movetime) return movetime;

	movetime = static_cast<uint64_t>(time[turn] / moves_to_go) + incr[turn];
	movetime -= movetime / 20 + incr[turn] / moves_to_go;

	return movetime;
}

void timemanager::set_movetime(const uint64_t new_time)
{
	movetime = new_time - new_time / 30;
}

void timer::start()
{
	start_time_ = std::chrono::system_clock::now();
}

uint64_t timer::elapsed() const
{
	return std::chrono::duration_cast<std::chrono::duration<uint64_t, std::ratio<1, 1000>>>
		(std::chrono::system_clock::now() - start_time_).count();
}
