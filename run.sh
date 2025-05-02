#!/bin/bash
clear

# Environment variables
export CUDA_LAUNCH_BLOCKING=1 
export TORCH_USE_CUDA_DSA=1

# Default settings
training_data=~/Datasets/tiny-shakespeare/train_coriolanus.csv.90percent
sample_model=tmp/trn_corio_adamw_e7_10percent.nlp_best.pt
num_tokens=256
test_data=~/Datasets/tiny-shakespeare/train_coriolanus.csv.10percent

# Create results directory if it doesn't exist
mkdir -p results

# Function to display the main menu
show_menu() {
  echo "=========================================="
  echo "       MinRNN Training and Evaluation     "
  echo "=========================================="
  echo "1. Train a model"
  echo "2. Sample from a model"
  echo "3. Compare models"
  echo "4. Cross-validate a model"
  echo "5. Batch cross-validate multiple models"
  echo "0. Exit"
  echo "=========================================="
  read -p "Enter your choice: " choice
}

# Function to train a model
train_model() {
  echo "=========================================="
  echo "            Training a Model              "
  echo "=========================================="
  read -p "Which architecture? [minGRU/minLSTM]: " arch
  read -p "Which optimizer? [adamw/sgd]: " the_optim
  read -p "Number of epochs [7]: " epochs
  epochs=${epochs:-7}
  read -p "Log with wandb? [0/1]: " log_online
  
  # Update settings.cfg with architecture
  sed -i "s/arch = .*/arch = $arch/" examples/settings.cfg
  sed -i "s/num_epochs = .*/num_epochs = $epochs/" examples/settings.cfg
  
  echo "Starting training with $arch architecture for $epochs epochs..."
  
  if [ "$log_online" -eq 1 ]; then
    echo 'Logging with wandb!'
    python -m examples.nlp_ds \
            train \
              --distributed \
              --wandb True \
              --optim "$the_optim" \
            $training_data
  else
    python -m examples.nlp_ds \
            train \
              --distributed \
              --optim "$the_optim" \
            $training_data
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
  
  if [ "$log_online" -eq 1 ]; then
    echo 'Logging with wandb!'
    python -m examples.nlp \
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

# Function to compare models
compare_models() {
  echo "=========================================="
  echo "           Model Comparison               "
  echo "=========================================="
  read -p "Enter glob pattern for models [tmp/nlp_best.epochs*.pt]: " model_pattern
  model_pattern=${model_pattern:-"tmp/nlp_best.epochs*.pt"}
  read -p "Use direct evaluation? [y/N]: " direct
  direct_flag=""
  if [[ "$direct" == "y" || "$direct" == "Y" ]]; then
    direct_flag="--direct"
  fi
  read -p "Sample size [$num_tokens]: " sample_size
  sample_size=${sample_size:-$num_tokens}
  
  echo "Comparing models matching $model_pattern..."
  python -m examples.model_comparison --models $model_pattern --sample-size $sample_size $direct_flag
}

# Function to cross-validate a model
cross_validate() {
  echo "=========================================="
  echo "          Cross-Validation                "
  echo "=========================================="
  read -p "Enter model path: " model_path
  read -p "Sample size [$num_tokens]: " sample_size
  sample_size=${sample_size:-$num_tokens}
  
  echo "Cross-validating model $model_path..."
  python -m examples.cross_validation $model_path --sample-size $sample_size
}

# Function to batch cross-validate models
batch_cross_validate() {
  echo "=========================================="
  echo "        Batch Cross-Validation            "
  echo "=========================================="
  read -p "Enter glob pattern for models [tmp/nlp_best.epochs*.pt]: " model_pattern
  model_pattern=${model_pattern:-"tmp/nlp_best.epochs*.pt"}
  read -p "Sample size [$num_tokens]: " sample_size
  sample_size=${sample_size:-$num_tokens}
  
  echo "Batch cross-validating models matching $model_pattern..."
  python -m examples.cross_validation --batch "$model_pattern" --sample-size $sample_size
}

# Main loop
while true; do
  show_menu
  
  case $choice in
    1) train_model ;;
    2) sample_model_fn ;;
    3) compare_models ;;
    4) cross_validate ;;
    5) batch_cross_validate ;;
    0) echo "Exiting..."; exit 0 ;;
    *) echo "Invalid option. Please try again." ;;
  esac
  
  echo
  read -p "Press Enter to continue..."
  #clear
done
