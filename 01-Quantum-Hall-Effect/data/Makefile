settings := 2K-20uA 2K-100uA 4K-20uA
targets := $(patsubst %,plots/%_main.pdf,$(settings))

.PHONY: data $(settings)

plots:
	mkdir plots

plots/%_main.pdf: % plot-hall.py plots
	./plot-hall.py $< --filter savgol --output-main $@

data: $(targets)

clean:
	git clean -Xf
