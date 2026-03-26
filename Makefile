# Assumptions
# - Python 3.13 in AutoROM directory
# - env.yml should end with dependencies for pip append
# - freeze's sed is still Linux syntax (not yet MacOS compatible)
# - one should still verify env.yml manually, freeze and env are over-engineered
ENV := ift702-docker

OS := $(shell uname)
ARCH := $(shell uname -m)

ifeq ($(OS),Darwin)
    ifeq ($(ARCH),arm64)
        PLATFORM := osx-arm64
    else
        PLATFORM := osx-64
    endif
endif

ifeq ($(OS),Linux)
    PLATFORM := linux-64
endif


### Storing current environment

freeze:
	conda env export --from-history -p $$CONDA_PREFIX \
	| grep -v -E "^(prefix|name):" > env.yml
	sed -i 's/- defaults/- conda-forge/' env.yml
	printf "  - pip\n  - pip:\n    - AutoROM\n" >> env.yml
	
lock: env.yml
	conda-lock lock -f env.yml -p $(PLATFORM)	

### Building environment

env: env.yml
	conda env create -n $(ENV) -f env.yml
	conda run -n $(ENV) AutoROM --accept-license \
	--install-dir "$$CONDA_PREFIX"/lib/python3.13/site-packages/ale_py/roms

unlock: conda-lock.yml
	conda-lock install -n $(ENV) conda-lock.yml
	conda run -n $(ENV) AutoROM --accept-license \
	--install-dir "$$CONDA_PREFIX"/lib/python3.13/site-packages/ale_py/roms

container :
	docker build -t ift702-docker .
	docker run --rm ift702-docker