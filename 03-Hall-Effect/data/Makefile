.PHONY: data

data: plots/mob-comp.pdf

plots:
	mkdir plots

constants.py: ../../common/constants.py
	cp $< $@

plots/mob-comp.pdf: eval-B.py src/data-A.dat src/data-B.dat plots
	./$< 2 --output $@

clean:
	git clean -Xf
