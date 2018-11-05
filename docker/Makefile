DOCKER_BUILD=docker build --rm=true

ifdef http_proxy
DOCKER_BUILD+=--build-arg http_proxy=$(http_proxy)
DOCKER_RUN_ENV+=--env "http_proxy=$(http_proxy)"
endif

ifdef https_proxy
DOCKER_BUILD+=--build-arg https_proxy=$(https_proxy)
DOCKER_RUN_ENV+=--env "https_proxy=$(https_proxy)"
endif

all: build_ngraph



build_ngraph: build_docker_image

	echo 'Building ngraph'


	docker run --rm --tty --interactive \
		${DOCKER_RUN_ENV} \
		build_ngraph_he \
		sh -c "./docker/build_ngraph_he_and_test.sh"


build_docker_image:
	./make_docker_image.sh