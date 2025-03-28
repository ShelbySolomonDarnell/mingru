clear

# Set up paths and parameters
test_data=~/Datasets/tiny-shakespeare/train_coriolanus.csv.90percent
sample_model=tmp/trn_corio_adamw_e7_10percent.nlp_best.pt
#sample_model=tmp/nlp_best.epochs7_minGRU_hidden256_512_1024_90percent_corio.pt
sample_size=128
num_samples=10
export CUDA_LAUNCH_BLOCKING=1 
export TORCH_USE_CUDA_DSA=1

# Install required package if missing
if ! python -c "import tiktoken" &> /dev/null; then
    echo "Installing tiktoken package..."
    pip install tiktoken
fi

# Run cross-validation
python -m examples.cross_validation \
    $sample_model $test_data \
    --sample-size $sample_size --wandb True
