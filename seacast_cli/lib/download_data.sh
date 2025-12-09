# This script is used to download data for Seacast CLI.
# It downloads reanalysis and ERA5 data, computes date splits, and prepares training, validation
# and test sets for the Atlantic dataset.
download_data() {
    # ----------------------------
    # Hyperparameters
    #
    # 0. Validación de argumentos
    ## Validar que se hayan pasado los argumentos necesarios
    while getopts "u:p:s:e:" opt; do
        case $opt in
            u) user="$OPTARG" ;;
            p) password="$OPTARG" ;;
            s) start_date="$OPTARG" ;;
            e) end_date="$OPTARG" ;;
            *) echo "Uso (yyyy-mm-dd): $0 -u user -p password -s start_date -e end_date"; exit 1 ;;
        esac
    done

    ## Validar que todas estén definidas
    if [ -z "${user:-}" ] || [ -z "${password:-}" ] || [ -z "${start_date:-}" ] || [ -z "${end_date:-}" ]; then
        echo "Faltan argumentos. Uso: $0 -u user -p password -s start_date -e end_date"
        exit 1
    fi

    local PYENV_PATH
    source ~/Seacast/seacast_cli/config/download_data.env

    # ----------------------------
    # DATA DOWNLOAD
    # ----------------------------
    # CONFIGURATION SUMMARY
    printf "%s\n" "=================================================="
    printf "%s\n" "       Data Download Configuration Summary        "
    printf "%s\n" "=================================================="
    printf "%s\n" "User:                 ${user}"
    printf "%s\n" "Password:             ·······"
    printf "%s\n" "Start date:           ${start_date}"
    printf "%s\n" "End date:             ${end_date}"
    printf "%s\n" "=================================================="

    local start_time=$(date +%s)
    local start_datetime=$(date +"%Y-%m-%d %H:%M:%S")
    printf "%s\n" ":: [${start_datetime}]: 🚀 Starting downloading"

    # 1. Data Download
    printf "%s\n" ":::: [$(date +"%Y-%m-%d %H:%M:%S")]: 📥 Downloading marine data from CMEMS..."
    ## Download marine data from CMEMS
    if ${PYENV_PATH}/python ~/Seacast/download_data.py \
                        --static \
                        -b data/atlantic/ \
                        -u "${user}" \
                        -psw "${password}"
    then
        printf "%s\n" ":::: [$(date +"%Y-%m-%d %H:%M:%S")]: ✅ Marine data downloaded successfully."
    else
        printf "%s\n" ":::: [$(date +"%Y-%m-%d %H:%M:%S")]: ❌ Failed to download marine data."
        exit 1
    fi

    ## Download reanalysis and ERA5 data     
    printf "%s\n" "[$(date +"%Y-%m-%d %H:%M:%S")]: 📥 Downloading reanalysis and ERA5 data..."              
    if ${PYENV_PATH}/python ~/Seacast/download_data.py \
                        -d reanalysis \
                        -s ${start_date} \
                        -e ${end_date} \
                        -u ${user} \
                        -psw ${password}
    ${PYENV_PATH}/python ~/Seacast/download_data.py \
                        -d era5 \
                        -s ${start_date} \
                        -e ${end_date} 
    then
        printf "%s\n" ":::: [$(date +"%Y-%m-%d %H:%M:%S")]: ✅ Reanalysis and ERA5 data downloaded successfully."
    else
        printf "%s\n" ":::: [$(date +"%Y-%m-%d %H:%M:%S")]: ❌ Failed to download reanalysis and ERA5 data."
        exit 1
    fi

    # 2. Data Preparation
    printf "%s\n" "::: [$(date +"%Y-%m-%d %H:%M:%S")]: ✂️ Computing date splits..."
    ## Llamada al helper Python que calcula los splits
    read train_start train_end val_start val_end test_start test_end < <(
    ${PYENV_PATH}/python ~/Seacast/compute_date_splits.py \
        --start_date ${start_date} --end_date ${end_date}
    )

    # 3. Prepare States - Reanalysis Data
    printf "%s\n" ":::: [$(date +"%Y-%m-%d %H:%M:%S")]: 📦📦📦 Preparing training, validation, and test sets for reanalysis data..."
    ## Training set
    if ${PYENV_PATH}/python ~/Seacast/prepare_states.py \
                        -d data/atlantic/raw/reanalysis \
                        -o data/atlantic/samples/train \
                        -n 6 \
                        -p rea_data \
                        -s ${train_start} \
                        -e ${train_end}

        # Validation set
    ${PYENV_PATH}/python ~/Seacast/prepare_states.py \
                        -d data/atlantic/raw/reanalysis \
                        -o data/atlantic/samples/val \
                        -n 6 \
                        -p rea_data \
                        -s ${val_start} \
                        -e ${val_end}

        # Test set
    ${PYENV_PATH}/python ~/Seacast/prepare_states.py \
                    -d data/atlantic/raw/reanalysis \
                    -o data/atlantic/samples/test \
                    -n 17 \
                    -p rea_data \
                    -s ${test_start} \
                    -e ${test_end} 
    then    
        printf "%s\n" ":::: ✅ Train: ${train_start} → ${train_end}"
        printf "%s\n" ":::: ✅ Val:   ${val_start} → ${val_end}"
        printf "%s\n" ":::: ✅ Test:  ${test_start} → ${test_end}"
    else
        printf "%s\n" ":::: [$(date +"%Y-%m-%d %H:%M:%S")]: ❌ Failed to prepare sets (train, validation, test)."
        exit 1
    fi


    # 4. Prepare States - ERA5 Data (forcing)
    ## Note: The ERA5 data is used as forcing data
    printf "%s\n" ":::: [$(date +"%Y-%m-%d %H:%M:%S")]: 📦📦📦 Preparing training, validation, and test sets for ERA5 data (forcing)..."
    # Training set
    if ${PYENV_PATH}/python ~/Seacast/prepare_states.py \
                        -d data/atlantic/raw/era5 \
                        -o data/atlantic/samples/train \
                        -n 6 \
                        -p forcing \
                        -s ${train_start} \
                        -e ${train_end}

    # Validation set
    ${PYENV_PATH}/python ~/Seacast/prepare_states.py \
                        -d data/atlantic/raw/era5 \
                        -o data/atlantic/samples/val \
                        -n 6 \
                        -p forcing \
                        -s ${val_start} \
                        -e ${val_end}
    
    # Test set
    ${PYENV_PATH}/python ~/Seacast/prepare_states.py \
                        -d data/atlantic/raw/era5 \
                        -o data/atlantic/samples/test \
                        -n 17 \
                        -p forcing \
                        -s ${test_start} \
                        -e ${test_end}
    then
        printf "%s\n" ":::: [$(date +"%Y-%m-%d %H:%M:%S")]: ✅ ERA5 datasets (train, validation, test) prepared successfully."
    else
        printf "%s\n" ":::: [$(date +"%Y-%m-%d %H:%M:%S")]: ❌ Failed to prepare ERA5 datasets (train, validation, test)."
        exit 1
    fi

    # 5. Feature and Model Preparation
    printf "%s\n" ":::: [$(date +"%Y-%m-%d %H:%M:%S")]: 🧩 Preparing features and model for the Atlantic dataset..."
    # Create grid features
    ${PYENV_PATH}/python ~/Seacast/create_grid_features.py --dataset atlantic

    # Create parameter weights
    ${PYENV_PATH}/python ~/Seacast/create_parameter_weights.py --dataset atlantic --batch_size 4 --n_workers 4

    # Create mesh
    ${PYENV_PATH}/python ~/Seacast/create_mesh.py --dataset atlantic --graph hierarchical --levels 3 --hierarchical 1
    printf "%s\n" ":::: [$(date +"%Y-%m-%d %H:%M:%S")]: ✅ Features and model prepared successfully."

    local end_time=$(date +%s)
    local end_datetime=$(date +"%Y-%m-%d %H:%M:%S")
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    printf "%s\n" ":::: [${end_datetime}]: 🎉 Data download and preparation completed in ${minutes} min ${seconds} sec (${duration} seconds)"
    # ----------------------------
    local summary=$(cat <<EOF
==================================================
                    Final Summary
==================================================
User: ${user}
Password: ·······
Start date: ${start_date}
End date: ${end_date}
Train set: ${train_start} → ${train_end}
Validation set: ${val_start} → ${val_end}
Test set: ${test_start} → ${test_end}
==================================================
EOF
)
    printf "%s\n" "${summary}"
    printf "%s\n" "${summary}" > ~/Seacast/reports/download_data_summary.txt
}

