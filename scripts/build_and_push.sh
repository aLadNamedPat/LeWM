#!/bin/bash
# Build and push Docker image to Docker Hub for RunPod deployment

set -e

# Configuration
IMAGE_NAME="le-world-model"
DOCKER_USERNAME="${DOCKER_USERNAME:-your-dockerhub-username}"
TAG="${TAG:-latest}"

echo "Building Docker image: ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"

# Build the image
docker build -t ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG} .

# Tag with version if provided
if [ ! -z "$VERSION" ]; then
    docker tag ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG} ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}
fi

echo "Pushing to Docker Hub..."
docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}

if [ ! -z "$VERSION" ]; then
    docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}
fi

echo "✅ Docker image pushed successfully!"
echo "Image: ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"
