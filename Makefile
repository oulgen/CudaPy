GHC_VERSION = $(shell ghc --version | grep -o '[[:digit:]].*$$')
translator = py2cuda

all: translator pylib

translator: cabal_hack
	# Build
	cd ${translator} && cabal configure --enable-shared && cabal build
	# Copy binary files
	mkdir -p bin
	# cp ${translator}/dist/build/libHS${translator}*.a bin/${translator}.a
	cp ${translator}/dist/build/libHS${translator}*.dylib bin/${translator}.so

cabal_hack:
	# We have to manually link the runtime library in cabal for some reason.
	# We also need to specify the GHC version correctly. This replaces the
	# 'extra-libraries' field in the cabal file with the correct version of rts.
	sed -i.tmp 's/\(extra-libraries:.*HSrts-ghc\).*/\1${GHC_VERSION}/g' ${translator}/${translator}.cabal

pylib: translator
	nvcc -O2 --shared --compiler-options '-fPIC' -o bin/libCudaPy.so cudapy/libCudaPy.cu

	# Generate an egg file
	mkdir -p dist
	mkdir -p dist/cudapy

	cp cudapy/*.py dist/cudapy/
	mv dist/cudapy/setup.py dist/
	cp bin/${translator}.so dist/cudapy/
	cp bin/libCudaPy.so dist/cudapy/

	cd dist && python setup.py bdist_egg
	cp dist/dist/*.egg bin/cudapy.egg

	rm -rf dist

clean:
	cd ${translator} && cabal clean && rm -rf dist
	rm -rf bin
