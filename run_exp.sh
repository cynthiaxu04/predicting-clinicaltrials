#!/bin/bash
BASE_DIR="$(pwd)"

FILES=(
    "$BASE_DIR/data/phase1_oncology_2011_api.json" 
    "$BASE_DIR/data/phase2_oncology_2011_api.json" 
    "$BASE_DIR/data/phase3_oncology_2011_api.csv")

BINS=(2 3)

MODEL_ARGS=("rf" "xgb" "lgb")

PROCESS_SCRIPT="$BASE_DIR/data/processing.py"
TRAIN_SCRIPT="$BASE_DIR/model/train_model.py"
TEST_SCRIPT="$BASE_DIR/model/get_test_results.py"

for FILE in "${FILES[@]}"; do
    if [[ "$FILE" == *.json ]]; then
        FILE_ARG="--json"
    elif [[ "$FILE" == *.csv ]]; then
        FILE_ARG="--csv"
    else
        echo "Unsupported file type: $FILE"
        continue
    fi
    #echo "Using file: $FILE"

    for BIN in "${BINS[@]}"; do
        echo "Running $PROCESS_SCRIPT with data file $FILE and integer $BIN"
        python3 "$PROCESS_SCRIPT" $FILE_ARG "$FILE" --bins "$BIN"

        for MODEL_ARG in "${MODEL_ARGS[@]}"; do
            echo "Training model $MODEL_ARG with $TRAIN_SCRIPT"
            python3 "$TRAIN_SCRIPT" "$MODEL_ARG"

            echo "Evaluating model $MODEL_ARG with $TEST_SCRIPT"
            python3 "$TEST_SCRIPT" "$MODEL_ARG"
        done
    done
done
