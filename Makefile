autoformat:
	black dcbench/ tests/
	autoflake --in-place --remove-all-unused-imports -r dcbench	tests
	isort --atomic dcbench/ tests/
	docformatter --in-place --recursive dcbench tests

lint:
	isort -c dcbench/ tests/
	black dcbench/ tests/ --check
	flake8 dcbench/ tests/

test:
	pytest

test-basic:
	set -e
	python -c "import dcbench as mk"
	python -c "import dcbench.version as mversion"

test-cov:
	pytest --cov=./ --cov-report=xml

docs:
	sphinx-build -b html docs/source/ docs/build/html/

docs-check:
	sphinx-build -b html docs/source/ docs/build/html/ -W

livedocs:
	sphinx-autobuild -b html docs/source/ docs/build/html/

dev:
	pip install black isort flake8 docformatter pytest-cov sphinx-rtd-theme nbsphinx recommonmark pre-commit

all: autoformat lint docs test
