.PHONY: all docs format

all: docs format


docs: 
	$(MAKE) -C docs html

format:
	docformatter -i -r IMLCV
	yapf -i -r IMLCV
