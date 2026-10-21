#!/bin/bash
echo "Starting EMG Pipeline..."

# Clean all previous results
echo "Cleaning previous results..."
rm -rf outputs/ state/ logs/
mkdir -p outputs/BLOCK_0 state logs

# Run each block
echo "Running BLOCK_0..."
python BLOCK_0_OPT1.py --quick || exit 1

echo "Running BLOCK_1..."
python BLOCK_1_PRE1.py || exit 1

echo "Running BLOCK_2..."
python BLOCK_2_FE1.py || exit 1

echo "Running BLOCK_3..."
python BLOCK_3_ML1.py || exit 1

echo "✅ Pipeline completed successfully!"
