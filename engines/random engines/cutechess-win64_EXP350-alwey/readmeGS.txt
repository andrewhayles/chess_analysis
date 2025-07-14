CuteChess GUI + Cli Custom build version EXPER-3.5.0 compiled 2022-01-30

get source here => https://github.com/alwey/cutechess/commits/EXPER-3
(This is a cutechess fork branch for testing and adding new code before
it is merged into the official cutechess repo)

If you already have another former CuteChessGUI installed please make
a copy of your engines.json and cutechess.ini from $user/appdata/roaming/cutechess
(hidden folder) for safety reasons.

Changes:

- adapted colors of my board design to evalhistory graph => gui/src/evalhistory.cpp

................................................
97	auto cWhite = QColor(0xff, 0xce, 0x9e);
98	auto cBlack = QColor(0xd1, 0x8b, 0x47);
................................................
	auto cWhite = QColor(0xec, 0xcb, 0xa2);
	auto cBlack = QColor(0xbb, 0x81, 0x56);
................................................

- time saved in pgn with almost minimum precision 2 digits => lib/src/chessgame.cpp

...............................
43	int precision = 0;
44	if (t < 100)
45		precision = 3;
46	else if (t < 1000)
47		precision = 2;
48	else if (t < 10000)
49		precision = 1;
...............................
	int precision = 2;
	if (t < 100)
		precision = 3;
	else if (t > 100000)
		precision = 1;
...............................		

Regensburg, 2022-04-02