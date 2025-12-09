# Inference script for running model inference on a specified graph
# Usage: ./inference.sh -c <ckpt_name>
# Example: ./inference.sh -c "hi_lam-4x128-07_02_00-0622-bathymetry_14_5"
# This script runs inference on a specified checkpoint and graph, and moves the prediction files to the output directory.
# It also generates a summary of the inference process.
inference() {
    # ----------------------------
    # Hyperparameters
    #
    # 0. Argument validation
    ## validate that the necessary arguments are passed
    # Default values
    sanity_check="false"
    while getopts "s:c:d:" opt; do
      case $opt in
        s) sanity_check="$OPTARG" ;;
        c) ckpt_name="$OPTARG" ;;
        d) device="$OPTARG" ;;
        *) echo "Uso: $0 [-s sanity_check] -c ckpt_name: e.g. hi_lam-4x128-07_02_00-0622-bathymetry_14_5 -d device: [cuda, cpu]"; exit 1 ;;
      esac
    done

    # Validate that all parameters are defined (except for sanity_check)
    if [ -z "${ckpt_name:-}" ] || [ -z "${device:-}" ]; then
      echo "Faltan argumentos. Uso: $0 [-s sanity_check] -c ckpt_name: e.g. hi_lam-4x128-07_02_00-0622-bathymetry_14_5 -d device: [cuda, cpu]"
      exit 1
    fi

    local PYENV_PATH CKPT_PATH OUTPUT_PATH TEST_PATH
    source ~/Seacast/seacast_cli/config/inference.env
    source ~/Seacast/seacast_cli/lib/seacast_utils.sh

    # ----------------------------
    # RUN INFERENCES
    # ----------------------------
    local graph_name="${ckpt_name##*-}"
    # ----------------------------
    # CONFIGURATION SUMMARY
    printf "%s\n" "=================================================="
    printf "%s\n" "           Inference Configuration Summary        "
    printf "%s\n" "=================================================="
    printf "%s\n" "Checkpoint/model name: ${ckpt_name}"
    printf "%s\n" "Device:                ${device}"
    printf "%s\n" "Graph name extracted:  ${graph_name}"
    printf "%s\n" "Checkpoint path:       ${CKPT_PATH}/${ckpt_name}/last.ckpt"
    printf "%s\n" "Output path:           ${OUTPUT_PATH}/${graph_name}/predictions"
    printf "%s\n" "=================================================="

    local start_time=$(date +%s)
    local start_datetime=$(date +"%Y-%m-%d %H:%M:%S")

    # If sanity check is enabled, create a temporary test folder
    if [ "${sanity_check}" == "true" ]; then
        printf "%s\n" ":::: 🩺 Sanity check enabled. Creating temporary test"
        # Actually, the count must be 10 because we copy both forcing and rea_data files
        five_paths=$(first_five "${TEST_PATH}")
        count=$(echo "${five_paths}" | wc -l) 
        # -----------------------------
        # CREATE TEMPORARY TEST FOLDER
        # -----------------------------
        # Define new test path: data/atlantic/samples/test_real
        new_path="${TEST_PATH/test/test_real}"
        # Prepare sanity check by renaming and copying files
        printf "%s\n" ":::: [$(date +"%Y-%m-%d %H:%M:%S")]: 🩺 Preparing sanity check by renaming and copying files ..."
        prepare_sanity_check "${TEST_PATH}" "${new_path}" "${five_paths[@]}"
    fi

        
    printf "%s\n" ":: [${start_datetime}]: 🚀 Starting inference: ${ckpt_name}"
    printf "%s\n" ":::: [$(date +"%Y-%m-%d %H:%M:%S")]: 🔮 Running inference with model ${ckpt_name} on graph ${graph_name}"
    # --data_subset reanalysis Actually, this is the satellite data
    ${PYENV_PATH}/python ~/Seacast/train_model.py \
                        --dataset atlantic \
                        --data_subset reanalysis \
                        --n_workers 16 \
                        --n_nodes 1 \
                        --batch_size 1 \
                        --step_length 1 \
                        --model hi_lam \
                        --graph $graph_name \
                        --processor_layers 4 \
                        --hidden_dim 128 \
                        --n_example_pred 1 \
                        --store_pred 1 \
                        --eval test \
                        --eval_device "${device}" \
                        --precision bf16-mixed \
                        --load "${CKPT_PATH}/${ckpt_name}/last.ckpt" \
                        --custom_run_name "pred_${graph_name}" && \
    printf "%s\n" ":::: [$(date +"%Y-%m-%d %H:%M:%S")]: ✅ Inference completed for model ${ckpt_name} on graph ${graph_name}"
    
    # If sanity check is enabled, restore the original test folder
    if [ "${sanity_check}" == "true" ]; then
        printf "%s\n" ":::: 🩺 Sanity check enabled"
        # -----------------------------
        # RESTORE ORIGINAL TEST FOLDER
        # -----------------------------
        # Remove temporary test folder: data/atlantic/samples/test
        printf "%s\n" ":::: [$(date +"%Y-%m-%d %H:%M:%S")]: 🩺 Restoring original test folder: ${TEST_PATH} ..."
        restore_test_folder ${TEST_PATH} ${new_path} ${count}
    fi
    
    # Copy predictions to output directory
    local SOURCE=${HOME}/Seacast/wandb/latest-run/files/predictions
    local TARGET=${OUTPUT_PATH}/${graph_name}/predictions
    move_prediction_files "${SOURCE}" "${TARGET}"
    # Compute scores for the current model checkpoint
    printf "%s\n" ":::: [$(date +"%Y-%m-%d %H:%M:%S")]: 🥇 Scoring predictions for model ${ckpt_name}"
    ${PYENV_PATH}/python ~/Seacast/score_model.py \
                        --pred_dir ${TARGET} \
                        --test_dir ${TEST_PATH} \
                        --variables "sst_temperature" && \
    printf "%s\n" ":::: [$(date +"%Y-%m-%d %H:%M:%S")]: ✅ Scoring completed for model ${ckpt_name}"
    local end_time=$(date +%s)
    local end_datetime=$(date +"%Y-%m-%d %H:%M:%S")
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))                    
    printf "%s\n" ":::: [${end_datetime}]: 🎉 Inference and scoring completed in ${minutes} min ${seconds} sec (${duration} seconds)"
    # ----------------------------
    local summary=$(cat <<EOF
==================================================
                    Final Summary
==================================================
Checkpoint/model used: ${ckpt_name}
Device: ${device}
Graph name: ${graph_name}
Checkpoint path: ${CKPT_PATH}/${ckpt_name}/last.ckpt
Predictions source directory: ${SOURCE}
Predictions target directory: ${TARGET}
Inference completed at: ${end_datetime}
Inference duration: ${minutes} min ${seconds} sec (${duration} seconds)
==================================================
In the target folder you'll find the copied predictions.
==================================================
EOF
)
    printf "%s\n" "${summary}"
    printf "%s\n" "${summary}" > "${OUTPUT_PATH}/${graph_name}/inference_summary.txt"
}