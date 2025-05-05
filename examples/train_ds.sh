#!/bin/bash
clear

# Environment variables
export CUDA_LAUNCH_BLOCKING=1 
export TORCH_USE_CUDA_DSA=1

# Default settings
sample_model=/home/shelbys/code/werernns/mingru/tmp/trn_corio_adamw_e7_10percent.nlp_best.pt
num_tokens=256

# Create results directory if it doesn't exist
mkdir -p ../results
# Create results/model directory if it doesn't exist
mkdir -p ../results/model

# Function to display the main menu
show_menu() {
  echo "=========================================="
  echo "       MinRNN Training and Evaluation     "
  echo "=========================================="
  echo "1. Train a model"
  echo "2. Sample from a model"
  echo "0. Exit"
  echo "=========================================="
  read -p "Enter your choice: " choice
}

# Function to train a model
train_model() {
  echo "=========================================="
  echo "            Training a Model              "
  echo "=========================================="
  read -p "Number of epochs [7]: " epochs
  epochs=${epochs:-7}
  read -p "Log with wandb? [0/1]: " log_online
  
  # Update settings.cfg with architecture
  #sed -i "s/arch = .*/arch = $arch/" settings.cfg
  sed -i "s/num_epochs = .*/num_epochs = $epochs/" settings.cfg
  
  #echo "Starting training with $arch architecture for $epochs epochs..."
  
  if [ "$log_online" -eq 1 ]; then
    echo 'Logging with wandb!'
    deepspeed --bind_cores_to_rank nlp_ds.py \
              train \
              --wandb True\ 
              --local_rank -1 
  else
    deepspeed --bind_cores_to_rank nlp_ds.py \
              train \
              --local_rank -1 
  fi
}

# Function to sample from a model
sample_model_fn() {
  echo "=========================================="
  echo "          Sampling from a Model           "
  echo "=========================================="
  read -p "Enter model path [$sample_model]: " model_path
  model_path=${model_path:-$sample_model}
  read -p "Enter prompt text: " sampletxt
  read -p "Number of tokens [$num_tokens]: " tokens
  tokens=${tokens:-$num_tokens}
  read -p "Log with wandb? [0/1]: " log_online
  cd ..
  
  if [ "$log_online" -eq 1 ]; then
    echo 'Logging with wandb!'
    python -m examples.nlp_ds \
            sample \
              --wandb True \
              --precond "$sampletxt" \
              --num-tokens $tokens \
            $model_path
  else
    python -m examples.nlp \
            sample \
              --precond "$sampletxt" \
              --num-tokens $tokens \
            $model_path
  fi
}

# Main loop
while true; do
  show_menu
  
  case $choice in
    1) train_model ;;
    2) sample_model_fn ;;
    0) echo "Exiting..."; exit 0 ;;
    *) echo "Invalid option. Please try again." ;;
  esac
  
  echo
  read -p "Press Enter to continue..."
  #clear
done