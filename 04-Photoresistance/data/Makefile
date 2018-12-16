.PHONY: data

data: plots/IV-549.pdf  plots/IV-647.pdf  plots/IV-dark-cap.pdf  plots/IV-dark-no-cap.pdf plots/IV-all.pdf plots/wavelength-current.pdf plots/frequency-current.pdf plots/intensity-current.pdf

plots:
	mkdir plots

plots/IV-all.pdf: plot.py plots
	./plot.py --output $@ --grid --xlabel U V --ylabel I mA \
		src/IV-549.dat --label "549 nm" - \
		src/IV-647.dat --label "647 nm" - \
		src/IV-dark-cap.dat --label "with cap" - \
		src/IV-dark-no-cap.dat --label "lab background"

plots/IV-%.pdf: src/IV-%.dat plot.py plots
	./plot.py $< --output $@ --grid --xlabel U V --ylabel I mA --fit $(if $(filter %549.pdf, $@),--fit-ignore 1)

plots/intensity-current.pdf: src/Intensity-Current-5V-549.dat plot.py plots
	./plot.py $< --output $@ --grid --xlabel I/I_0 "" --ylabel I mA

plots/wavelength-current.pdf: src/Wavelength-Current-5V.dat plot.py plots
	./plot.py $< --output $@ --grid --xlabel \\lambda nm --ylabel I mA --fit-gaussian

plots/frequency-current.pdf: src/Frequency-Current.dat plot.py plots
	./plot.py $< --output $@ --grid --xlabel f Hz --ylabel I "$$\\mu\\mathrm{A_{RMS}}$$" --fit-freq-shitty 10 15 --logx --logy

clean:
	git clean -Xf
