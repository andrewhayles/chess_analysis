#pragma once
#include <sstream>
#include <thread>

#include "board.h"
#include "timemanager.h"

namespace uci
{
	void go(std::istringstream& stream, std::thread& searching, board& pos, timemanager& chrono);
	bool ignore(std::thread& searching);
	void loop();
	void options();
	void position(std::istringstream& stream, board& pos);
	void search(board* pos, timemanager* chrono);
	void setoption(std::istringstream& stream);
	void stop_thread_if(std::thread& searching);
	void infinite(timemanager& chrono);
	uint32_t to_move(const board& pos, std::string input);
}
