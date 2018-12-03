.PHONY: data

data: plots/IV-549.pdf  plots/IV-647.pdf  plots/IV-dark-cap.pdf  plots/IV-dark-no-cap.pdf plots/IV-all.pdf

plots:
	mkdir plots

plots/IV-all.pdf: plot.py plots
	./plot.py --output $@ --grid --xlabel U V --ylabel I mA src/IV-549.dat --label "549 nm" - src/IV-647.dat --label "647 nm" - src/IV-dark-cap.dat --label "with cap" - src/IV-dark-no-cap.dat --label "lab background"
plots/IV-%.pdf: src/IV-%.dat plot.py plots
	./plot.py $< --output $@ --grid --xlabel U V --ylabel I mA

clean:
	git clean -Xf
