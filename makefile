.PHONY: all docs format

all: docs format

docs: 
	$(MAKE) -C docs html

format:
	docformatter -i -r --wrap-summaries 120 --wrap-descriptions 120  IMLCV
	yapf -i -r --style='{based_on_style:google, COLUMN_LIMIT:120}' IMLCV
