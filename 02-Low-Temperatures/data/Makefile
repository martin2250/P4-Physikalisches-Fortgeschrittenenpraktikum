.PHONY: data

data: plots/Bc2-over-T.pdf plots/lns-over-inv-T.pdf plots/overview.pdf plots/overview-mag.pdf plots/separate.pdf plots/separate-mag.pdf

plots:
	mkdir plots

constants.py: ../../common/constants.py
	cp $< $@

plots/Bc2-over-T.pdf: Bc2-over-T.py constants.py plots
	./$< --output $@

plots/lns-over-inv-T.pdf: lns-over-inv-T.py constants.py plots
	./$< --output $@

plots/overview.pdf: R-over-T.py loadRT.py constants.py plots
	./$< overview --output $@

plots/overview-mag.pdf: R-over-T.py loadRT.py constants.py plots
	./$< overview --magnify --output $@

plots/separate.pdf: R-over-T.py loadRT.py constants.py plots
	./$< separate --output $@

plots/separate-mag.pdf: R-over-T.py loadRT.py constants.py plots
	./$< separate --magnify --output $@

clean:
	git clean -Xf
