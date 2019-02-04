DOCKER_BUILD=docker build --rm=true

ifdef http_proxy
DOCKER_BUILD+=--build-arg http_proxy=$(http_proxy)
DOCKER_RUN_ENV+=--env "http_proxy=$(http_proxy)"
endif

ifdef https_proxy
DOCKER_BUILD+=--build-arg https_proxy=$(https_proxy)
DOCKER_RUN_ENV+=--env "https_proxy=$(https_proxy)"
endif

GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD)

all: build_and_test_he_transformer


build_and_test_he_transformer: build_docker_image
	echo 'Building and testing he-transformer'
	echo 'Git branch $(GIT_BRANCH)'

	docker run --rm --tty --interactive \
		${DOCKER_RUN_ENV} \
		he_transformer \
		sh -c "git clone https://github.com/NervanaSystems/he-transformer.git && \
					 cd he-transformer && \
					 git checkout $(GIT_BRANCH) && \
					 chmod +x docker/build_and_test_he_transformer.sh && \
					 docker/build_and_test_he_transformer.sh"

build_docker_image:
	./make_docker_image.sh