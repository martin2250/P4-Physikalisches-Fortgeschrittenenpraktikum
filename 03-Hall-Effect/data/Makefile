.PHONY: data

data: plots/mob-comp.pdf plots/arrhenius.pdf plots/n.pdf plots/sample-a-conductivity.pdf plots/sample-a-product.pdf plots/table-a.agtex plots/table-b.agtex

plots:
	mkdir plots

constants.py: ../../common/constants.py
	cp $< $@

plots/mob-comp.pdf: eval-B.py src/data-A.dat src/data-B.dat plots
	./$< 2 --output $@

plots/sample-a-conductivity.pdf:  eval-A.py src/data-A.dat constants.py plots
	./$< 2.1 --output $@ --size 5 4

plots/sample-a-product.pdf:  eval-A.py constants.py src/data-A.dat plots
	./$< 2.2 --output $@  --size 5 4

plots/n.pdf: eval-A.py src/data-A.dat constants.py plots
	./$< 4 --output $@  --size 5 4

plots/arrhenius.pdf: eval-A.py src/data-A.dat constants.py plots
	./$< 5+6 --output $@ --size 4.5 3.5

plots/table-a.agtex: eval-B.py src/data-A.dat plots
	./$< 1 --table-A $@

plots/table-b.agtex: eval-B.py src/data-B.dat plots
	./$< 1 --table-B $@

clean:
	git clean -Xf
