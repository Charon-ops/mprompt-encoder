#!/bin/bash

# port cuda
ports=(11434 11435 11436 11437)
cudas=(0 1 2 3)

length=${#ports[@]}

# close screen
for ((i=0; i<$length; i++)); do
    name="ollama${cudas[$i]}"
    echo "close screen $name"
    screen -S $name -X quit
done

# start screen
for ((i=0; i<$length; i++)); do
    name="ollama${cudas[$i]}"
    port="127.0.0.1:${ports[$i]}"
    cuda="${cudas[$i]}"
    screen -dmS $name
    screen -x -S $name -p 0 -X stuff "conda activate mprompt \r"
    screen -x -S $name -p 0 -X stuff "export OLLAMA_HOST=$port \r"
    screen -x -S $name -p 0 -X stuff "export CUDA_VISIBLE_DEVICES=$cuda \r"
    screen -x -S $name -p 0 -X stuff "export OLLAMA_NUM_PARALLEL=1 \r"
    screen -x -S $name -p 0 -X stuff "export OLLAMA_MAX_LOADED_MODELS=4 \r"
    screen -x -S $name -p 0 -X stuff "ollama serve \r"

    echo "start screen $name, port:$port, cuda:$cuda"
    
done