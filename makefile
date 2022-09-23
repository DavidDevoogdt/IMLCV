.PHONY: all docs format test

all: docs format test

docs: 
	$(MAKE) -C docs html

format:
	docformatter -i -r --wrap-summaries 79 --wrap-descriptions 79  .
	yapf -i -r --style='{based_on_style:google, COLUMN_LIMIT:79,indent_width: 4}' .
test:
	pytest

clean:
	rm -rf IMLCV/.parsl_scripts
	rm -rf IMLCV/.runinfo
	rm -rf IMLCV/.ase_calculators
	rm -rf IMLCV/.bash_python_app
	rm -rf IMLCV/test/output/hpc_perovskite
	rm -rf IMLCV/test/output/test_cv_disc_perov
	