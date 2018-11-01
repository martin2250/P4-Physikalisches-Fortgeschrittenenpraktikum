settings := 2K-20uA 2K-100uA 4K-20uA
targets := $(patsubst %,plots/%-main.pdf,$(settings))

.PHONY: data $(settings)

data: $(targets) plots/2K-20uA-hw.pdf

plots:
	mkdir plots

plots/%-main.pdf: % plot-hall.py plots
	./plot-hall.py $< --filter savgol --output-main $@

plots/2K-20uA-hw.pdf: plot-haywire.py
	./plot-haywire.py 2K-20uA -o $@

clean:
	git clean -Xf
