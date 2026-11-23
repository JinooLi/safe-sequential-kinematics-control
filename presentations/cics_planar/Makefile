.PHONY: all slides presenter notes clean
all: notes slides presenter
notes:
	python3 scripts/export_notes.py
slides:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex
	cp build/main.pdf main.pdf
presenter:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build presenter.tex
	cp build/presenter.pdf presenter.pdf
clean:
	latexmk -c -outdir=build main.tex presenter.tex
