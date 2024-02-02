#!/bin/bash

# Define variables for image and container
IMAGE="noahbruderer/baio:0.0.1"
CONTAINER_NAME="baio"

# Function to stop the container
function stop_container {
    echo "Stopping and removing the Docker container..."
    /usr/local/bin/docker stop $CONTAINER_NAME
}

# Pull the latest version of the image
/usr/local/bin/docker pull $IMAGE

# Start the Docker container with a specific name and the --rm flag
/usr/local/bin/docker run --name $CONTAINER_NAME --rm -p 8501:8501 $IMAGE &

/usr/local/bin/docker logs -f $CONTAINER_NAME

# Wait a bit to ensure the service is up
sleep 3

# Open the browser
open http://localhost:8501

# Wait for user input to stop the container
while true; do
    read -p "Type 'stop' to stop the container: " input
    if [ "$input" = "stop" ]; then
        stop_container
        break
    fi
done
