#include "engine.h"
#include "uci.h"

int main()
{
	std::cout << eng_name << " " << version << " " << platform << std::endl;
	engine::init_eval();
	engine::init_magic();
	engine::init_movegen();
	uci::loop();
	return 0;
}
