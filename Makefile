docker-build:
	docker build -t probcomp/gen-quickstart:v0 .
.PHONY: docker-build

docker-build-tagged:
	docker build -t probcomp/gen-quickstart:$(shell git rev-parse --short HEAD) .
.PHONY: docker-build-tagged
