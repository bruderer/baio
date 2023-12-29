#!/bin/bash


#use for dev and test

# Rebuild the image and start the Docker container using docker-compose
docker-compose up --build -d

# Wait a bit to ensure the service is up
sleep 3

# Open the browser to the service
open http://localhost:8501

# Stream the logs of the container
echo "Streaming container logs. Press Ctrl+C to stop."
docker-compose logs -f

# Note: The script will continue to follow the logs until you press Ctrl+C.
# After stopping the log stream, you can manually stop the container with:
# docker-compose down
