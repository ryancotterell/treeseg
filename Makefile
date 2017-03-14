all: cython

cython:
	python setup.py build_ext -i

install: cython
	pip install -e .

clean:
	rm -f *~
	find -name '*.so' | xargs rm -rf
	rm -rf build
	rm -rf dist
	rm -rf segmenter.egg-info
	rm -rf .coverage coverage-report
	find -name '*.pyc' -exec rm {} \;
	find -name '*.so' -exec rm {} \;
	find -name '*.html' -exec rm {} \;
	find -name '*.cpp' -exec rm {} \;
	find -name '*.c' -exec rm {} \;
