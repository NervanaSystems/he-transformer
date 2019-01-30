DOCKER_BUILD=docker build --rm=true

ifdef http_proxy
DOCKER_BUILD+=--build-arg http_proxy=$(http_proxy)
DOCKER_RUN_ENV+=--env "http_proxy=$(http_proxy)"
endif

ifdef https_proxy
DOCKER_BUILD+=--build-arg https_proxy=$(https_proxy)
DOCKER_RUN_ENV+=--env "https_proxy=$(https_proxy)"
endif

all: build_and_test_he_transformer


build_and_test_he_transformer: build_docker_image
	echo 'Building and testing he-transformer'

	docker run --rm --tty --interactive \
		${DOCKER_RUN_ENV} \
		build_ngraph_he \
		sh -c "pwd && ls && ./build_and_test_he_transformer.sh"


build_docker_image:
	./make_docker_image.sh