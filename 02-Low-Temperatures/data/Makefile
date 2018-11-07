.PHONY: data

data: constants.py

plots:
	mkdir plots

constants.py: ../../common/constants.py
	cp $< $@

plots/Bc2-over-T.pdf: Bc2-over-T.py
	./$< --output $@

clean:
	git clean -Xf
