# Docker Builds for he-transformer with a _Reference-OS_

## Introduction

This directory contains a basic build system for creating docker images of the _reference-OS_ on which ngraph builds and unit tests are run.  The purpose is to provide reference builds for _Continuous Integration_ used in developing and testing ngraph.

The `Makefile` provides targets for:

* Building the _reference-OS_ into a docker image
* Building he-transformer and running unit tests in this cloned repo, mounted into the docker image of the _reference-OS_

The _make_ targets are designed to handle all aspects of building the _reference-OS_ docker image, running ngraph builds and unit testing in it, and opening up a session in the docker image for interactive use.  You should not need to issue any manual commands (unless you want to).  In addition the `Dockerfile.ngraph.*` files provide a description of how each _reference-OS_ environment is built, should you want to build your own server or docker image.

## Prerequisites

In order to use the _make_ targets, you will need to do the following:

* Have *docker* installed on your computer with the docker daemon running.
* These scripts assume that you are able to run the `docker` command without using `sudo`.  You will need to add your account to the `docker` group so this is possible.
* If your computer (running docker) sits behind a firewall, you will need to have the docker daemon properly configured to use proxies to get through the firewall, so that public docker registries and git repositories can be accessed.

## Make Targets

The _make_ targets are designed to provide easy commands to run actions using the docker image.  All _make_ targets should be issued on the host OS, and _not_ in a docker image.

* Running the command **`make build_docker_image`** will create the docker image from the reference OS.
* Running the command **`make build_and_test_he_transformer`** will build he-transformer and run a series of unit-tests.

## Examples/Hints

* To build an Ubuntu 16.04 docker container, compile with gcc 5.4, and run unit tests:

```
cd contrib/docker
make check_gcc
```

* To build an Ubuntu 16.04 docker container, compile with clang 3.9, and run unit tests:

```
cd contrib/docker
make check_clang
```

* To build a CentOS 7.4 docker container, compile with gcc 4.8.5, and run unit tests:

```
cd contrib/docker
make check_gcc OS=centos74
```

## Helper Scripts

These helper scripts are included for use in the `Makefile`.  **These scripts should _not_ be called directly unless you understand what they do.**

#### `build_and_test_he_transformer.sh`

A helper script to simplify the building and testing of he-transformer using docker images.

#### `make_docker_image.sh`

A helper script to simplify building of docker images for the reference OS environment.

## Notes
* The docker image curently obtains the source code from the host. Hence, any changes in your local codebase will be reflected in the Docker image.