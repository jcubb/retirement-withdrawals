.PHONY: all clean wipe

all: paper.pdf notation.pdf

paper.pdf: paper.tex
	latexmk -pdf paper.tex

notation.pdf: notation.tex
	latexmk -pdf notation.tex

clean:
	latexmk -c paper.tex notation.tex

wipe:
	latexmk -C paper.tex notation.tex
