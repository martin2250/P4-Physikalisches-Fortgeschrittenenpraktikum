BUILD := latexmk -pdf -interaction=batchmode -synctex=1 -file-line-error
BUILDV := latexmk -pdf -interaction=nonstopmode -synctex=1 -file-line-error
FILE := $(shell basename "$(CURDIR)")
svgimgs := $(wildcard img/*.svg)
svgimtargets := $(patsubst %.svg,%.pdf,$(svgimgs))

ifeq ($(VERBOSE), 1)
	BUILD := $(BUILDV)
endif

.PHONY: all show upload share clean data build hooks

all: | $(FILE).pdf show hooks

build: $(FILE).pdf hooks

data: hooks
	make -C data -j 6

$(FILE).pdf: data $(FILE).tex ../common/emails.tex bibliography.ag.tex titlepage.ag.tex $(svgimtargets) preamble.fmt
	$(BUILD) $(FILE).tex

preamble.fmt: preamble.tex
	pdflatex -ini -jobname="preamble" "&pdflatex preamble.tex\dump"

show: hooks
	xdg-open $(FILE).pdf > /dev/null 2>&1 &

share: build
	curl -H "Max-Days: 1" --upload-file ./$(FILE).pdf https://transfer.sh/$(FILE).pdf
	@echo

../common/emails.tex:
	printf '\\newcommand{\\emailA}{TODO: umut@dick.butt}' > ../common/emails.tex
	printf '\\newcommand{\\emailB}{TODO: martin@dick.butt}' >> ../common/emails.tex

img/%.pdf: img/%.svg
	inkscape $< --export-pdf=$@

#titlepage and bibliography have to be copied, because \include only works in subdirectories of $PWD
titlepage.ag.tex:
	cp ../praktikum-protokollvorlage-latex/include/titlepage_modern.tex titlepage.ag.tex
bibliography.ag.tex:
	cp ../common/bibliography.tex bibliography.ag.tex

clean:
	git clean -Xf

setup:
	../common/setup.sh "$(FILE).tex"

hooks: ../.git/hooks/commit-msg

../.git/hooks/commit-msg: ../common/hooks/commit-msg
	cp $< $@
