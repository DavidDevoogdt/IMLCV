.PHONY: all docs format test

all: docs format test

docs: 
	$(MAKE) -C docs html

format:
	docformatter -i -r --wrap-summaries 79 --wrap-descriptions 79  .
	yapf -i -r --style='{based_on_style:google, COLUMN_LIMIT:79,indent_width: 4}' .
test:
	pytest