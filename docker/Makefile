DOCKER_BUILD=docker build --rm=true

ifdef http_proxy
DOCKER_BUILD+=--build-arg http_proxy=$(http_proxy)
DOCKER_RUN_ENV+=--env "http_proxy=$(http_proxy)"
endif

ifdef https_proxy
DOCKER_BUILD+=--build-arg https_proxy=$(https_proxy)
DOCKER_RUN_ENV+=--env "https_proxy=$(https_proxy)"
endif

GIT_COMMIT = $(shell git rev-parse HEAD)


all: build_and_test_he_transformer

build_and_test_he_transformer: build_docker_image
	docker run --rm --tty --interactive \
		${DOCKER_RUN_ENV} \
		build_ngraph_he \
		sh -c "./docker/build_and_test_he_transformer.sh $(GIT_COMMIT)"

build_docker_image:
	./make_docker_image.sh