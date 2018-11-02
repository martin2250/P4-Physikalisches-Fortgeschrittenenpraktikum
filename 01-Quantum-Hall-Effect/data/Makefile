settings := 2K-20uA 2K-100uA 4K-20uA
targets := $(patsubst %,plots/%-main.pdf,$(settings))

.PHONY: data $(settings)

data: $(targets) plots/2K-20uA-hw.pdf plots/2K-20uA-unfiltered.pdf

plots:
	mkdir plots

plots/%-main.pdf: % plot-hall.py plots
	./plot-hall.py $< --filter savgol --output-main $@

plots/2K-20uA-hw.pdf: plot-haywire.py plots
	./plot-haywire.py 2K-20uA -o $@

plots/2K-20uA-unfiltered.pdf: plot-hall.py plots
	./plot-hall.py 2K-20uA --output-main $@

clean:
	git clean -Xf
