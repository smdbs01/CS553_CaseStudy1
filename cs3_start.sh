#!/bin/bash

# Command to build and run the docker image
# Run on the server
docker run -d --env-file .env -p 12345:7860 --name group9_cool_image_generator --network monitoring smdbs/cs553:cs3
