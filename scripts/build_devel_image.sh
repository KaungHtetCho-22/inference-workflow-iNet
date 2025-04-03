#!/bin/env bash

docker build \
	--build-arg="USERNAME=`(whoami)`" \
	--build-arg="USER_UID=`(id -u)`" \
	--build-arg="USER_GID=`(id -g)`" \
	--no-cache \
	-t bio-diversity-service:devel \
	--target devel \
	-f docker/Dockerfile .
