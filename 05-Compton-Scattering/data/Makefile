.PHONY: data

data: plots/diff-cross-sec.pdf plots/energy-shift.pdf

constants.py: ../../common/constants.py
	cp $< $@

plots:
	mkdir plots

plots/diff-cross-sec.pdf: diff-cross-section.py constants.py calibration.py datasets.py detector.py plots
	./$< $@

plots/energy-shift.pdf: energy-shift.py calibration.py datasets.py plots
	./$< $@

clean:
	git clean -Xf
