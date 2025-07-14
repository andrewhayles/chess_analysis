Black-Cat-Chess: default compilations on sse4.1/no pop hardware, which will run everywhere
Release versions: 1.2a, 2.0, 2.1, 2.2, 2.3
(note that I didn't compile 2.4, as it uses a SF nnue network, also neglected 1.0 because it had an early bugfix)

compiled: between 2022-07-26 and 2022-12-04 with GCC 12.1.0 in Msys2

sources: https://github.com/Checkmate6659/black-cat-chess

author: Alexander Wimmer (FRA)


From the release history:
*************************

Black Cat v2.3
--------------
01 Sep 09:45
@Checkmate6659 Checkmate6659
v2.3
3c651b5
Black Cat v2.3

Playing strength: ~2310elo +/- 14elo
Basic semi-UCI-compliance implemented (sufficient for playing chess games)

Added SEE pruning in main search and qsearch
Added Move Count Based Pruning/Late Move Pruning

WARNING: The engine can stall at very short time controls, but the stalls cannot be replicated

Has only been compiled for Linux (64-bit)


Black Cat v2.2
--------------
22 Aug 08:07
@Checkmate6659 Checkmate6659
v2.2
9af5028
Black Cat v2.2

Playing strength: ~2264elo
Basic semi-UCI-compliance implemented (sufficient for playing chess games)

Enhanced Null Move Pruning and Late Move Reductions
Added check extensions
Tuned engine

Has only been compiled for Linux (64-bit)


Black Cat v2.1
--------------
15 Aug 10:46
@Checkmate6659 Checkmate6659
v2.1
0b8ad1b
Black Cat v2.1

Playing Strength: ~1940elo
Basic semi-UCI-compliance implemented (sufficient for playing chess games)

Added null move pruning
Added tapered evaluation
Replaced Michinewski simplified eval with Rofchade/PeSTO PSQT eval

Has only been compiled for Linux


Black Cat v2.0
--------------
02 Aug 06:11
@Checkmate6659 Checkmate6659
v2.0
ae802e9
Black Cat v2.0

Playing Strength: ~1710elo
Basic semi-UCI-compliance implemented (sufficient for playing chess games)

Added transposition table
Added late move reduction
Fixed a few bugs

Has only been compiled for Linux


Black Cat v1.2a
---------------
25 Jul 07:01
@Checkmate6659 Checkmate6659
v1.2a
53d17a1
Black Cat v1.2a Pre-release

Playing Strength: ~1550elo
Basic semi-UCI-compliance implemented (sufficient for playing chess games)

Added killer and history heuristic
Fixed bug related to qsearch
Fixed UCI bug (search not stopping after keystroke)

Has only been compiled for Linux


Guenther Simon
Regensburg, 2022-12-04
