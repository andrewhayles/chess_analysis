# Project Title

> This project takes an input PGN and player name and analyzes a preset number of moves per game starting at a preset move number (at the top of the rating_estimate15.py script these parameters can be easily specified).  It produces an estimate of the player's rating based on the relative performance compared with many chess engines of different ratings their corresponding performance compared to an "oracle" engine, which is very strong and finds the "correct" move in each position.

## About The Project

I started this project after reading Hindemburg Melao Jr's book "Chess Two New Rating Systems: The 2022 Best Players in History (English Edition)"  He does a similar but more comprehensive analysis of tha games of many chess players so that he can create a ranking of the best players in history.  My methodology is similar in nature to his but it is not the same, and he has communicated to me that to get an accurate estimate of the rating one would need to use more chess engines.  I have been unable to acquire more reliable chess engines of various ratings easily online and so I have not proceeded to implement this change.

The problem this project solves is that the rating one receives (Elo) on chess.com and other chess sites can be misleading because of various reasons.  For example, if someone sets their mind to playing better, and starts taking their games seriously, their next games can be at a performance that is much higher than their previous games.  Their Elo will not reflect this though, because there has not been enough time to establish this change in their attitude towards chess.  So, this project provides a more granular approach to estimating rating, such that even just one game could be analyzed and someone's performance rating could be estimated based on several moves made in that one game.

### Built With

* Stockfish chess engine
* Leela chess engine with Maia weights
* Python

## Getting Started

If you download stockfish only it is possible to attempt an estimate using this project code.  If you download the Leela chess engine and you download the weights from the Maia chess github then you can do something closer to what was actually done in this project.  One problem with only using stockfish (even with rating limits) is that you may not get a wide enough range of performances (they all perform relatively well possibly, depending on the nature of the positions analyzed).  One indicated problem with the project as it is is that even with Maia engines with ratings as low as 1100, the move chosen by the engines are still relatively good, since the Maia engines were designed to pick the most human move based on the rating, which is often the move that is the most natural and can often coincide with the oracle move, such that even Maia 1100 scores quite well (50% hit rate or so) depending on the types of positions.

You'll also need the real_engines.csv file for the rating_estimate15.py code to run properly.  You'll need all of the directory paths properly specified.  

### Prerequisites

All of the associated libraries indicated in the rating_estimate15.py file are necessary.  It is recommended to set up a virtual environment to install all these chess libraries that are specialized and not normally needed.

### Installation

1.  Clone the repo
    ```sh
    git clone https://github.com/andrewhayles/chess_analysis.git
    ```
2.  Install packages
3.  Ensure rating_estimate15.py file names and player name are set and the PGN is in the same directory as the Python file etc.
4.  Keep in mind, the oracle cache and the goalgames.pgn are the analysis of my personal games.  A new oracle cache will be generated if you put a new pgn file in with the Python file and set the program to analyze that particular PGN file and delete my oracle cache or simply rename it.
5.  Run rating_estimate15.py


## Contact

Your Name - Andrew Hayles (andyhayles@gmail.com)

Project Link: [https://github.com/andrewhayles/chess_analysis](https://github.com/andrewhayles/chess_analysis)