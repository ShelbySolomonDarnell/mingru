clear

# Set up paths and parameters
test_data=~/Datasets/tiny-shakespeare/test.txt
sample_model=tmp/trn_corio_adamw_e7_10percent.nlp_best.pt
sample_size=32
num_samples=10
export CUDA_LAUNCH_BLOCKING=1 
export TORCH_USE_CUDA_DSA=1

# Install required package if missing
if ! python -c "import tiktoken" &> /dev/null; then
    echo "Installing tiktoken package..."
    pip install tiktoken
fi

# Run cross-validation
python examples/cross_validation.py \
    $sample_model \
    $test_data \
    --sample-size $sample_size \
    --num-samples $num_samples
