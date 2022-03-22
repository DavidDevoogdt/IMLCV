.PHONY: all docs format test

all: docs format test

docs: 
	$(MAKE) -C docs html

format:
	docformatter -i -r --wrap-summaries 120 --wrap-descriptions 120  .
	yapf -i -r --style='{based_on_style:google, COLUMN_LIMIT:120}' .
test:
	pytest