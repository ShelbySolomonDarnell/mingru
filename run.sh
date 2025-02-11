clear

#if [ ! -f logs/werernns.log ]; then
#  echo 'No file to remove.'
#else
#  rm logs/werernns.log
#fi

#CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 python testrnns.py
#CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 python trainrnns.py

#training_data=~/Datasets/tiny-shakespeare/train.csv
training_data=~/Datasets/tiny-shakespeare/train_coriolanus.csv
the_model=tmp/train.nlp_best.pt
sample_model=tmp/train_coriolanus.nlp_best.pt
num_tokens=256
export CUDA_LAUNCH_BLOCKING=1 
export TORCH_USE_CUDA_DSA=1

runCode()
{
  read -p "Enter 0 to train, 1 to train and use wandb logging, or and 2 to test, with 3 to test with wandb logging --> " testing
  if [ $testing -eq 0 ]; then
    echo 'Let us train!'
    python -m examples.nlp train $training_data
  elif [ $testing -eq 1 ]; then
    echo 'Let us train, and log with wandb!'
    python -m examples.nlp \
                train \
                  --wandb True \
                $training_data
  elif [ $testing -eq 2 ]; then 
    echo 'Let us test!'
    read -p "Type what you want the model to work with --> " sampletxt
    python -m examples.nlp \
                sample \
                  --precond "$sampletxt" \
                  --num-tokens $num_tokens \
                $sample_model
  else
    echo 'Let us test, and log with wandb!'
    read -p "Type what you want the model to work with --> " sampletxt
    python -m examples.nlp \ 
                sample 
                  --wandb True \
                  --num-tokens $num_tokens \
                  --precond "$sampletxt" \
                $sample_model
  fi
}

runCode
