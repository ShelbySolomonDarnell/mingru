clear

#if [ ! -f logs/werernns.log ]; then
#  echo 'No file to remove.'
#else
#  rm logs/werernns.log
#fi

#CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 python testrnns.py
#CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 python trainrnns.py

#training_data=~/Datasets/tiny-shakespeare/train.csv
training_data=~/Datasets/tiny-shakespeare/train_coriolanus.csv.90percent
#the_model=tmp/train_criolanus_90percent.nlp_best.pt
#sample_model=tmp/train_coriolanus.nlp_best.pt
sample_model=tmp/trn_corio_adamw_e7_10percent.nlp_best.pt
num_tokens=256
export CUDA_LAUNCH_BLOCKING=1 
export TORCH_USE_CUDA_DSA=1

runCode()
{
  read -p "Enter 0 to train, 1 to sample --> " testing
  if [ $testing -eq 0 ]; then
    echo 'Let us train!'
    read -p "Which optimizer should be used with the neural net type either [adamw or sgd] --> " the_optim
    read -p "Do you want to log online with wandb (True 1/False 0) --> " log_online
    if [ $log_online -eq 1 ]; then
      echo 'Logging with wandb!'
      python -m examples.nlp \
                  train \
                    --wandb True \
                    --optim "$the_optim" \
                  $training_data
    else
      python -m examples.nlp \
                  train \
                    --optim "$the_optim" \
                  $training_data
    fi
  elif [ $testing -eq 1 ]; then 
    echo 'Let us test!'
    read -p "Type what you want the model to work with --> " sampletxt
    read -p "Do you want to log online with wandb (False 0/True 1) --> " log_online
    if [ $log_online -eq 1 ]; then
      echo 'Logging with wandb!'
      python -m examples.nlp \
                  sample \
                    --wandb True \
                    --precond "$sampletxt" \
                    --num-tokens $num_tokens \
                  $sample_model
    else
      python -m examples.nlp \
                  sample \
                    --precond "$sampletxt" \
                    --num-tokens $num_tokens \
                  $sample_model
    fi
  fi
}

runCode
