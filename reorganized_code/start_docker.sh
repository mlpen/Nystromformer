docker run --ipc=host --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$1 -v "$PWD:/workspace" -it mlpen/transformers:4
