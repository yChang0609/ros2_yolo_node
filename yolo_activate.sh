#!/bin/bash

PORT_MAPPING=""
if [ "$1" = "--port" ] && [ -n "$2" ] && [ -n "$3" ]; then
    PORT_MAPPING="-p $2:$3"
    shift 3  # Remove the first three arguments
fi

# 檢查系統架構
ARCH=$(uname -m)
OS=$(uname -s)

if [ "$ARCH" = "aarch64" ]; then
    echo "Detected architecture: arm64"
    docker run -it --rm \
        --network compose_my_bridge_network \
        $PORT_MAPPING \
        --runtime=nvidia \
        --env-file .env \
        -v "$(pwd)/src:/workspace/src" \
        registry.screamtrumpet.csie.ncku.edu.tw/screamlab/jpack5_yolo_opencv_image:latest \
        /bin/bash
elif [ "$ARCH" = "x86_64" ] || { [ "$ARCH" = "arm64" ] && [ "$OS" = "Darwin" ]; }; then
    echo "Detected architecture: amd64 or macOS arm64"
    if [ "$OS" = "Darwin" ]; then
        # macOS 不使用 --gpus all
        docker run -it --rm \
            --network compose_my_bridge_network \
            $PORT_MAPPING \
            --env-file .env \
            -v "$(pwd)/src:/workspaces/src" \
            -v "$(pwd)/screenshots:/workspaces/screenshots" \
            registry.screamtrumpet.csie.ncku.edu.tw/screamlab/pros_cameraapi:0.0.2 \
            /bin/bash
    else
        # 檢查是否支援 --gpus all
        if docker run --help | grep -q "--gpus"; then
            docker run -it --rm \
                --network compose_my_bridge_network \
                $PORT_MAPPING \
                --gpus all \
                --env-file .env \
                -v "$(pwd)/src:/workspaces/src" \
                -v "$(pwd)/screenshots:/workspaces/screenshots" \
                registry.screamtrumpet.csie.ncku.edu.tw/screamlab/pros_cameraapi:0.0.2 \
                /bin/bash
        else
            echo "--gpus all not supported, running without it."
            docker run -it --rm \
                --network compose_my_bridge_network \
                $PORT_MAPPING \
                --env-file .env \
                -v "$(pwd)/src:/workspaces/src" \
                -v "$(pwd)/screenshots:/workspaces/screenshots" \
                registry.screamtrumpet.csie.ncku.edu.tw/screamlab/pros_cameraapi:0.0.2 \
                /bin/bash
        fi
    fi
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi
