#include "uci.h"

#include "bench.h"
#include "engine.h"

namespace uci
{
	void go(std::istringstream& stream, std::thread& searching, board& pos, timemanager& chrono)
	{
		std::string token;

		infinite(chrono);

		while (stream >> token)
		{
			if (token == "movestogo")
			{
				stream >> chrono.moves_to_go;
			}
			else if (token == "wtime")
			{
				stream >> chrono.time[white];
				chrono.movetime = 0;
			}
			else if (token == "btime")
			{
				stream >> chrono.time[black];
				chrono.movetime = 0;
			}
			else if (token == "winc")
			{
				stream >> chrono.incr[white];
			}
			else if (token == "binc")
			{
				stream >> chrono.incr[black];
			}
			else if (token == "ponder")
			{
				engine::infinite = true;
			}

			else if (token == "depth")
			{
				stream >> engine::depth;
				if (engine::depth > lim::depth)
					engine::depth = lim::depth;
			}
			else if (token == "nodes")
			{
				stream >> engine::nodes;
			}
			else if (token == "movetime")
			{
				stream >> token;
				chrono.set_movetime(stoi(token));
			}
			else if (token == "infinite")
			{
				engine::infinite = true;
			}
		}

		searching = std::thread{ search, &pos, &chrono };
	}

	void loop()
	{
		std::string input, token;
		board pos{};
		timemanager chrono;
		std::thread searching;

		engine::new_game(pos);

		do
		{
			std::getline(std::cin, input);

			std::istringstream stream(input);
			stream >> token;

			if (input == "uci")
			{
				options();
			}
			else if (input == "stop")
			{
				stop_thread_if(searching);
				engine::infinite = false;
			}
			else if (input == "ponderhit")
			{
				engine::stop_ponder();
			}
			else if (input == "isready")
			{
				std::cout << "readyok" << std::endl;
			}
			else if (input == "ucinewgame")
			{
				if (ignore(searching))
					continue;
				engine::new_game(pos);
			}
			else if (token == "setoption")
			{
				if (ignore(searching))
					continue;
				setoption(stream);
			}
			else if (token == "position")
			{
				if (ignore(searching))
					continue;
				position(stream, pos);
			}
			else if (token == "go")
			{
				if (ignore(searching))
					continue;
				go(stream, searching, pos, chrono);
			}
			else if (token == "perft")
			{
				if (!(stream >> token) || token == "legal" || token == "pseudolegal")
					bench::perft(token);
			}
		} while (input != "quit");

		stop_thread_if(searching);
	}

	void options()
	{
		std::cout << "id name" << " " << eng_name << " " << version << " " << platform << std::endl;
		std::cout << "id author" << " " << author << std::endl;
		std::cout << "option name Ponder type check default false" << std::endl;
		std::cout << "option name Hash type spin default " << engine::hash_size << " min 1 max " << lim::hash << std::endl;
		std::cout << "option name Clear Hash type button" << std::endl;
		std::cout << "option name Contempt type spin default " << engine::contempt << " min " << lim::min_cont << " max " <<
			lim::max_cont << std::endl;
		std::cout << "uciok" << std::endl;
	}

	void position(std::istringstream& stream, board& pos)
	{
		std::string token, fen;
		stream >> token;

		if (token == "startpos")
		{
			fen = engine::startpos;
			stream >> token;
		}
		else if (token == "fen")
		{
			while (stream >> token && token != "moves")
				fen += token + " ";
		}

		engine::parse_fen(pos, fen);

		if (token != "moves") return;

		while (stream >> token)
			engine::new_move(pos, to_move(pos, token));
	}

void search(board* pos, timemanager* chrono)
{
	engine::stop = false;

	uint32_t ponder{ 0 };
	const uint32_t best_move
	{
		engine::start_searching(*pos, *chrono, ponder)
	};

	assert(best_move != NO_MOVE);
	engine::stop = true;

	std::cout << "bestmove " << algebraic(best_move);

	if (ponder)
		std::cout << " ponder " << algebraic(ponder);
	std::cout << std::endl;
}

void setoption(std::istringstream& stream)
{
	std::string token, name, value;
	stream >> name, stream >> name, stream >> token;

	while (token != "value")
	{
		name += " " + token;
		if (!(stream >> token)) break;
	}
	stream >> value;

	if (name == "Hash")
	{
		const int new_hash{ stoi(value) };
		engine::new_hash_size(new_hash <= lim::hash ? new_hash : lim::hash);
	}
	else if (name == "Clear Hash")
	{
		engine::clear_hash();
	}
	else if (name == "Contempt")
	{
		int new_cont{ stoi(value) };

		if (new_cont < lim::min_cont) new_cont = lim::min_cont;
		if (new_cont > lim::max_cont) new_cont = lim::max_cont;

		engine::contempt = new_cont;
	}
	else if (name == "Ponder")
	{
	}
}

void stop_thread_if(std::thread& searching)
{
	if (searching.joinable())
	{
		engine::stop = true;
		searching.join();
	}
}

bool ignore(std::thread& searching)
{
	if (engine::stop)
	{
		if (searching.joinable())
			searching.join();
		return false;
	}
	return true;
}

void infinite(timemanager& chrono)
{
	engine::depth = lim::depth;
	engine::nodes = lim::nodes;
	chrono.movetime = lim::movetime;
	chrono.moves_to_go = 50;
}

uint32_t to_move(const board& pos, std::string input)
{
	const char promo{ input.back() };
	if (input.size() == 5) input.pop_back();

	assert(input.size() == 4);

	const auto sq1{ to_idx(input.substr(0, 2)) };
	const auto sq2{ to_idx(input.substr(2, 2)) };

	uint8_t flag{ no_piece };
	const auto piece{ pos.piece_sq[sq1] };
	auto victim{ pos.piece_sq[sq2] };

	if (piece == Pawn)
	{
		if (victim == no_piece && abs(sq1 - sq2) % 8 != 0)
		{
			flag = enpassant;
			victim = Pawn;
		}

		if (abs(sq1 - sq2) == 16)
			flag = doublepush;

		if (promo == 'q') flag = promo_queen;
		else if (promo == 'r') flag = promo_rook;
		else if (promo == 'b') flag = promo_bishop;
		else if (promo == 'n') flag = promo_knight;
	}

	else if (piece == King)
	{
		if (sq1 == e1 && sq2 == g1) flag = white_short;
		else if (sq1 == e8 && sq2 == g8) flag = black_short;
		else if (sq1 == e1 && sq2 == c1) flag = white_long;
		else if (sq1 == e8 && sq2 == c8) flag = black_long;
	}
	else
		assert(piece > PAWNS && piece < KINGS);

	return encode(sq1, sq2, flag, piece, victim, pos.turn);
}
}

