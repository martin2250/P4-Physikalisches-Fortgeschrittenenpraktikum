.PHONY: data

data: plots/Bc2-over-T.pdf plots/lns-over-inv-T.pdf

plots:
	mkdir plots

constants.py: ../../common/constants.py
	cp $< $@

plots/Bc2-over-T.pdf: Bc2-over-T.py constants.py
	./$< --output $@

plots/lns-over-inv-T.pdf: lns-over-inv-T.py constants.py
	./$< --output $@

clean:
	git clean -Xf
