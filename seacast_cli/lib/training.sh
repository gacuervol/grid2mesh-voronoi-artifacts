# Training script for hierarchical mesh generation and model training
# Usage: ./training.sh -e <epochs> -n <exp_name> -m <mesh_type> -o <mesh_nodes>
# Example: ./training.sh -e 100 -n "my_experiment" -m "bathymetry" -o "5,3"
training() {
    # ----------------------------
    # Hyperparameters
    #
    # 0. Argument validation
    ## validate that the necessary arguments are passed
    # Default values
    sanity_check="false"
    densification="fps"   # valor por defecto si no se pasa -d
    while getopts "s:e:n:m:o:c:x:d:" opt; do
      case $opt in
        s) sanity_check="$OPTARG" ;;
        e) epochs="$OPTARG" ;;
        n) exp_name="$OPTARG" ;;
        m) mesh_type="$OPTARG" ;;
        o) mesh_nodes="$OPTARG" ;;
        c) g2m_m2g_conect="$OPTARG" ;;
        x) crossing_edges="$OPTARG" ;;
        d) densification="$OPTARG" ;;
        *) echo "Uso: $0 [-s sanity_check] [-e epochs] [-n exp_name] [-m mesh_type]: e.g. [bathymetry, uniform, fps] [-o mesh_nodes]: e.g.'5, 3' [-c g2m_m2g_conect]: 1 -[x crossing_edges]: cross=1 non_cross=0 [-d densification]: e.g. [fps, fps_weighted]"; exit 1 ;;
      esac
    done

    # Validate that all parameters are defined (except for sanity_check)
    if [ -z "${epochs:-}" ] || [ -z "${exp_name:-}" ] || [ -z "${mesh_type:-}" ] || [ -z "${mesh_nodes:-}" ] || [ -z "${g2m_m2g_conect:-}" ] || [ -z "${crossing_edges:-}" ]; then
      echo "Faltan argumentos. Uso: $0 [-s sanity_check] [-e epochs] [-n exp_name] [-m mesh_type]: e.g. [bathymetry, uniform, fps] [-o mesh_nodes]: e.g.'5, 3' [-c g2m_m2g_conect]: 1 -[x crossing_edges]: cross=1 non_cross=0"
      exit 1
    fi

    # If sanity check is enabled, set epochs to 1
    if [ "${sanity_check}" == "true" ]; then
        epochs=1
        exp_name="sanity_check_${exp_name}"
        printf "%s\n" ":::: 🩺 Sanity check enabled. Setting epochs to ${epochs}."
    fi
    # Define the script directory
    local SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    # Load environment variables from the config file
    local PYENV_PATH SCRIPT_PATH MESH_PATH PLOTS_PATH OUTPUT_PATH
    source "${SCRIPT_DIR}/../config/train.env"
    # Load utility functions
    source "${SCRIPT_DIR}/../lib/seacast_utils.sh"

    printf "%s\n" ${OUTPUT_PATH}
    # Ensure the output directory exists
    if [ ! -d "$(eval echo "${OUTPUT_PATH}")" ]; then
        mkdir -p "$(eval echo "${OUTPUT_PATH}")"
    fi

    # Print the configuration summary
    printf "%s\n" "============================================="
    printf "%s\n" "      Experiment configuration summary:      "
    printf "%s\n" "============================================="
    printf "%s\n" "Experiment name: ${exp_name}"
    printf "%s\n" "Number of epochs: ${epochs}"
    printf "%s\n" "Mesh type: ${mesh_type}"
    printf "%s\n" "Number of nodes in the mesh: ${mesh_nodes}"
    printf "%s\n" "G2M-M2G connections: ${g2m_m2g_conect}"
    printf "%s\n" "Densification method: ${densification} (if mesh_type is fps or bathymetry)"
    printf "%s\n" "============================================="

    # EXPERIMENT
    # ----------------------------
    # Get the current date and time for logging
    local start_time=$(date +%s)
    local start_datetime=$(date +"%Y-%m-%d %H:%M:%S")
    local mesh_nodes_format=$(format_mesh_nodes "${mesh_nodes}") # Format mesh nodes: "5, 4, 3" -> "5_4_3"
    run_name="${exp_name}_m_${mesh_nodes_format}_g2mm2g_${g2m_m2g_conect}"
    printf "%s\n" ":: [${start_datetime}]: 🚀 Starting experiment: ${run_name}"
    ## Create a hierarchical mesh
    printf "%s\n" ":::: [$(date +"%Y-%m-%d %H:%M:%S")]: ⏳ Creating hierarchical mesh with nodes: ${mesh_nodes_format}"
    
    if [ "${mesh_type}" == "bathymetry" ]; then
        printf "%s\n" ":: bathymetry mesh type selected"
        ${PYENV_PATH}python ${SCRIPT_PATH}create_non_uniform_mesh.py \
            --dataset atlantic \
            --plot 1 \
            --mesh_type ${mesh_type} \
            --levels 2 \
            --crossing_edges 1 \
            --nodes_amount "${mesh_nodes}" \
            --n_connections ${g2m_m2g_conect} \
            --k_neighboors 1
    elif [ "${mesh_type}" == "uniform" ]; then
        printf "%s\n" ":: uniform mesh type selected"
        ${PYENV_PATH}python ${SCRIPT_PATH}create_non_uniform_mesh.py \
            --dataset atlantic \
            --plot 1 \
            --mesh_type ${mesh_type} \
            --levels 2 \
            --crossing_edges "${crossing_edges}" \
            --uniform_resolution_list "${mesh_nodes}" \
            --n_connections ${g2m_m2g_conect} \
            --k_neighboors 1
    elif [ "${mesh_type}" == "fps" ]; then
        printf "%s\n" ":: fps mesh type selected"
        ${PYENV_PATH}python ${SCRIPT_PATH}create_non_uniform_mesh.py \
            --dataset atlantic \
            --plot 1 \
            --mesh_type ${mesh_type} \
            --levels 2 \
            --crossing_edges "${crossing_edges}" \
            --nodes_amount "${mesh_nodes}" \
            --n_connections ${g2m_m2g_conect} \
            --k_neighboors 1 \
            --sampler ${densification}
    else
      echo "❌ Error: Unknown mesh_type '${mesh_type}'. Must be 'bathymetry' 'uniform' or 'fps'." >&2
      exit 1
    fi

    ## Move files to the correct directory
    printf "%s\n" ":::: [$(date +"%Y-%m-%d %H:%M:%S")]: ⏳ Moving files to the correct directory"
    ${PYENV_PATH}python ${SCRIPT_PATH}move_files.py --graph_type hierarchical --graph ${mesh_type}
    ## Train the model
    printf "%s\n" ":::: [$(date +"%Y-%m-%d %H:%M:%S")]: 🎓 Training model with ${mesh_type} edges and nodes: ${mesh_nodes_format}"
    ${PYENV_PATH}python ~/Seacast/train_model.py \
                        --dataset atlantic \
                        --n_nodes 1 \
                        --n_workers 16 \
                        --epochs ${epochs} \
                        --lr 0.001 \
                        --batch_size 1 \
                        --step_length 1 \
                        --ar_steps 1 \
                        --optimizer adamw \
                        --scheduler cosine \
                        --processor_layers 4 \
                        --hidden_dim 128 \
                        --model hi_lam \
                        --graph hierarchical \
                        --custom_run_name ${run_name} \
                        --finetune_start 1

    local end_time=$(date +%s)
    local end_datetime=$(date +"%Y-%m-%d %H:%M:%S")
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    printf "%s\n" ":::: [${end_datetime}]: 🎉 Training completed"
    # ----------------------------
    # MOVE OUTPUT FILES
    #-----------------------------
    # This function moves output files from the mesh and plots directories to the target directory.
    local EXP_DIR="${OUTPUT_PATH}/${run_name}"
    move_output_files "${EXP_DIR}" "${mesh_type}" "${MESH_PATH}" "${PLOTS_PATH}" || {
      printf "%s\n" "Error moving output files"
      exit 1
    }
    # ----------------------------
    # COPY GRAPH FILES
    #-----------------------------
    # This function copies graph files from the output directory to the target directory.
    local SOURCE="${OUTPUT_PATH}/${run_name}"
    local TARGET="${HOME}/Seacast/graphs/${run_name}"
    move_graph_files "${SOURCE}" "${TARGET}"
    # ----------------------------

    local summary=$(cat <<EOF
==================================================
                    Final Summary
==================================================
Experiment name: ${exp_name}
Number of epochs: ${epochs}
Mesh type: ${mesh_type}
Number of nodes in the mesh: ${mesh_nodes}
G2M-M2G connections: ${g2m_m2g_conect}
Start date and time: ${start_datetime}
End date and time: ${end_datetime}
Total training duration: ${minutes} min ${seconds} sec (${duration} seconds)

Generated files and their location:
- Results and trained models: ${EXP_DIR}
- Copied graph files: ${TARGET}

In the results folder, you'll find the generated models, logs, and plots.
In the graph folder, you'll find the files used for analysis and visualization.
==================================================
EOF
)
    printf "%s\n" "${summary}"
    printf "%s\n" "${summary}" > "${EXP_DIR}/summary.txt"
}