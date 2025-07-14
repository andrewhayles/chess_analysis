                        +-------------------------------+
                        |  Prédateur 2.2.1 Readme file  |
                        |     February 8th, 2012        |
                        +-------------------------------+

Prédateur is a UCI compatible chess engine programmed by Julien MARCEL
( jul_marcel@hotmail.com , http://predateur-chess.blogspot.com/ ).

If you like this engine and/or just want to show the author your support,
please consider donating a few bucks using the PayPal button on the blog. :o)


Foreword:
---------

Because of constant pressure from Ryxa, Leo and Graham, and because the x32 version of
Prédateur 2.2 was consistantly outplayed by Fernando Villega's Moron 1.0, I had no choice
but to release this Prédateur 2.2.1 bugfix version.

I couldn't help but to temper with other things here and there, so its strength might be
slightly different (more in the downside than in the upside).

(For some reasons, and because of the hurry, the Mac versions are a bit slow...)

Have fun with it and thank you all for your patience! :)



I - What changed since last update ?
------------------------------------
- fixed the ponder mode bug 
- fixed the time bug 
- x32 version works again 
- corrected a PST that mistakenly wasn't identical for both sides 
- a few tweaks here and there 
- Prédateur doesn't stop searching when it finds a mate anymore. That way it keeps searching while it still has time and maybe find a shorter one. 
- made all uci options lowercase (does CuteChess still protest?)
- new binary: MacOSX PPC

II - How to run it ?
--------------------
Prédateur is "just" a chess engine, i.e. it is a brain with no legs nor arms. You'll need a graphical chess interface to run it, one that handles the UCI protocol (Universal Chess Interface). My personal choise is the great Arena interface : it is free and has too many options and functions for you to test them all in your whole life.
But Prédateur will also work well in Chessbase's interfaces (Fritz,..), Shredder's and even under Winboard using Fabien Letouzay's Polyglot adaptor.

In arena, just select "add new engine" in the engine menu and find Prédateur's drawer in your harddisk (though it is strongly suggested to create a "Prédateur" drawer in Arena's "engine" drawer). You can then change a few options in Prédateur's "option" pannel in Arena, like the size of the hash tables.

WINDOWS:
--------
Here is a list of chess interfaces you can use with Prédateur on Windows:
- Arena (http://www.playwitharena.com)
- ChessGUI (http://www.chessgui.com/)
- ChessDB (Scid) (http://prolinux.free.fr/scid)
- Mayura Chess (http://www.mayura.com/chess)
- Winboard (using Polyglot, the uci->wb adapter)
- And of course some commercial products like ChessBase's, Fritz's or Shredder's interfaces.

MAC:
----
On Mac, Prédateur will run with most of interfaces, like:
- Sigma Chess (http://www.sigmachess.com/) (doesn't work with Lion, as it's a PPC interface)
- ExaChess (http://www.exachess.com)
- Jose (http://jose-chess.sourceforge.net/index_download.html)
- Probably Shredder's interface too

On MorphOS, there is no UCI compatible chess interface (except maybe Jose if you manage to make it run), so you'll have to run Prédateur in a shell and use UCI commands (read more about this protocole here : ), for instance:
> ucinewgame
> go movetime 10000
(10000 is a time in milliseconds).
Hopefully someone will get motivated enough to program a UCI compatible interface for the AmigaOSes...


III - Thanks and aknowledgements
--------------------------------
- The FreePascal and Lazarus teams, for providing us with a wonderful and
  state of the art development tool : thanks a lot!
- The Arena team, for giving us such a great chess interface!
- David Levy, for his inspiring books about chess computers
- Robert Hyatt, for pioneering
- Bruce Moreland, Ed Shröder, Colin Frayn, Jonatan Pettersson, Scott Gasch and Verhelst
  for their chess webpages
- All the testers, for their time and passion : Olivier, Leo, Graham, Adam, ...
- Leo Dijksman and Patrick Buchmann for their priceless job with
  computer chess news (thanks!)
- the Talkchess and Winboard forum teams and participants