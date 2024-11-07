#!/bin/bash

PORT_MAPPING=""
if [ "$1" = "--port" ] && [ -n "$2" ] && [ -n "$3" ]; then
    PORT_MAPPING="-p $2:$3"
    shift 3  # Remove the first three arguments
fi

# 修改 docker run 命令以掛載 yoloy_detect.py
docker run -it --rm \
    --network compose_my_bridge_network \
    $PORT_MAPPING \
    --runtime=nvidia \
    --env-file .env \
    -v "$(pwd)/src:/workspace/src" \
    yolo_ros2:latest \
    /bin/bash

    