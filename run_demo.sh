#!/bin/sh

docker run --net fakenews --ip 69.69.69.3 -p 6969:6969 flask-server &
docker run --net fakenews --ip 69.69.69.2 -p 8000:8000 api-gateway

cd ~/repos/website && npm start