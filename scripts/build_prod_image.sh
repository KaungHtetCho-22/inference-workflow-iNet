#!/bin/env bash

docker build \
	-t bio-diverstiy-service:prod \
	--target prod \
	-f docker/Dockerfile .
