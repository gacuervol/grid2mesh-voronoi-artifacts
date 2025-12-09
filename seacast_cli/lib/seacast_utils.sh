# Mueve archivos de grafo (*.pt) desde un directorio fuente a uno destino
# This function moves graph files from a source directory to a destination directory.
# Parameters:
#   - SOURCE: The source directory containing the graph files.
#   - DEST: The destination directory where the graph files will be moved.
# Returns:
#   - 0 on success, 1 on failure.
# Usage:
#   move_graph_files "/path/to/source" "/path/to/destination"
move_graph_files() {
    local SOURCE="$1"
    local DEST="$2"

    if [[ -z "${SOURCE}" || -z "${DEST}" ]]; then
        printf "%s\n" "❌ Usage: move_graph_files <source_dir> <dest_dir>"
        return 1
    fi

    printf "%s\n" ":::: [$(date +"%Y-%m-%d %H:%M:%S")]: ⏳ Moving graph files from ${SOURCE} to ${DEST}"

    mkdir -p "${DEST}" && \
    cp -f "${SOURCE}"/*.pt "${DEST}" && \
    printf "%s\n" ":::: [$(date +"%Y-%m-%d %H:%M:%S")]: ✅ Graph files moved to ${DEST}"
    }

# Move output files to the target directory
# This function uses rsync to move files from the source mesh and plots directories to the target directory.
# It removes the source files after moving them to save space.
# It also removes the source directories after all files have been moved.
# Parameters:
#   - TARGET_DIR: The directory where the files will be moved.
#   - MESH_TYPE: The type of mesh (e.g., "bathymetry").
#   - MESH_PATH: The path to the mesh files.
#   - PLOTS_PATH: The path to the plots files.
# Returns:
#   - 0 on success, 1 on failure.
# Usage:
#   move_output_files "/path/to/target" "bathymetry" "/path/to/mesh" "/path/to/plots"
move_output_files() {
    local TARGET_DIR="$1"
    local MESH_TYPE="$2"
    local MESH_PATH="$3"
    local PLOTS_PATH="$4"

    printf "%s\n" ":::: [$(date +"%Y-%m-%d %H:%M:%S")]: ⏳ Moving mesh files to ${TARGET_DIR}"
    mkdir -p "${TARGET_DIR}" || {
        printf "%s\n" "Error: no se pudo crear el directorio ${TARGET_DIR}"
        return 1
    }

    rsync -a --remove-source-files "${MESH_PATH}/${MESH_TYPE}/" "${TARGET_DIR}/" && rm -r "${MESH_PATH}" || {
        printf "%s\n" "Error moviendo archivos de $MESH_PATH"
        return 1
    }

    rsync -a --remove-source-files "${PLOTS_PATH}${MESH_TYPE}/" "${TARGET_DIR}/" && rm -r "${PLOTS_PATH}" || {
        printf "%s\n" "Error moviendo archivos de $PLOTS_PATH"
        return 1
    }

    printf "%s\n" ":::: [$(date +"%Y-%m-%d %H:%M:%S")]: ✅ Output files moved from ${MESH_PATH} and ${PLOTS_PATH} to ${OUTPUT_PATH}/${exp_name}_${mesh_nodes_format}/"
}

# Move prediction files from source to target directory
# This function copies prediction files from the source directory to the target directory.
# It creates the target directory if it does not exist.
# Parameters:
#   - SOURCE: The source directory containing the prediction files.
#   - TARGET: The target directory where the prediction files will be copied.
# Returns:
#   - 0 on success, 1 on failure.
# Usage:
#   move_prediction_files "/path/to/source" "/path/to/target"
move_prediction_files() {
    local SOURCE="$1"
    local TARGET="$2"

    if [[ -z "${SOURCE}" || -z "${TARGET}" ]]; then
        printf "%s\n" "❌ Usage: move_prediction_files <source_dir> <target_dir>"
        return 1
    fi

    mkdir -p "${TARGET}"

    if cp -r "${SOURCE}/." "${TARGET}/"; then
        printf "%s\n" ":::: [$(date +"%Y-%m-%d %H:%M:%S")]: 📂 Predictions copied from ${SOURCE} to ${TARGET}"
    else
        printf "%s\n" ":::: [$(date +"%Y-%m-%d %H:%M:%S")]: ❌ Failed to copy predictions from ${SOURCE} to ${TARGET}"
        return 1
    fi
}

# Format mesh nodes for output directory naming
# This function formats the mesh nodes by removing spaces and replacing commas with underscores.
# Parameters:
#   - mesh_nodes: A string containing the mesh nodes, e.g., "node1, node2, node3".
# Returns:
#   - A formatted string suitable for use in directory names, e.g., "node1_node2_node3".
# Usage:
#   formatted_nodes=$(format_mesh_nodes "node1, node2, node3")
#   echo "$formatted_nodes"  # Output: "node1_node2_node3"
format_mesh_nodes() {
  local mesh_nodes="$1"
  local formatted_nodes
  formatted_nodes=$(echo "${mesh_nodes}" | tr -d ' ' | tr ',' '_')
  echo "$formatted_nodes"
}

# Get the first five forcing and rea_data files from the test path
# This function lists the first five forcing files and their corresponding rea_data files (10 files).
# Parameters:
#   - test_path: The path to the test directory containing forcing files.
# Returns:
#   - A list of the first five forcing files and their corresponding rea_data files.
# Usage:
#   five_paths=$(first_five "/path/to/test")
first_five() {
    local test_path="$1"
    for forcing_path in $(ls ${test_path}/forcing_*.npy | head -n 5); do
        echo "$forcing_path"
        # ${variable/patrón/reemplazo}
        rea_path="${forcing_path/forcing_/rea_data_}"
        echo "$rea_path"
    done
}

# Prepare the test folder for a sanity check
# This function renames the original test folder, creates a temporary test folder, and copies the first five files from the original test folder to the temporary folder.
# Parameters:
#   - test_path: The path to the original test folder.
#   - backup_path: The path to the backup folder where the original test folder will be renamed.
#   - npy_paths: An array of paths to the first five files to be copied.
# Returns:
#   - None. It modifies the file system by renaming and copying files.
# Usage:
#   prepare_sanity_check "/path/to/test" "/path/to/test_real" "${npy_paths[@]}"
prepare_sanity_check() {
    local test_path="$1"
    local backup_path="$2"
    shift 2
    local npy_paths=("$@")

    # Rename: data/atlantic/samples/test -> data/atlantic/samples/test_real
    printf "%s\n" ":::: Renaming original test folder: ${test_path} to ${backup_path} ..."
    mv "${test_path}" "${backup_path}"
    # Create empty temporary test folder: data/atlantic/samples/test
    printf "%s\n" ":::: Creating empty temporary test folder: ${test_path} ..."
    mkdir -p "${test_path}"
    # Copy first five files from test_real to test
    printf "%s\n" ":::: Copying first five files from ${backup_path} to ${test_path} ..."
    for npy in ${npy_paths}; do
        cp "${npy/$test_path/$backup_path}" "${test_path}"
        printf "%s\n" ":::: 📨 Copied: ${npy/$test_path/$backup_path} to ${test_path}"
    done
}

# Restore the original test folder after a sanity check
# This function checks if the temporary test folder contains exactly 10 files (5 forcing and 5 rea_data).
# If the count is correct, it restores the original test folder; otherwise, it exits with an error.
# Parameters:
#   - test_path: The path to the temporary test folder.
#   - backup_path: The path to the backup folder where the original test folder was renamed.
#   - expected_count: The expected number of files in the temporary test folder (should be 10).
# Returns:
#   - None. It modifies the file system by restoring the original test folder or exiting with an error.
# Usage:
#   restore_test_folder "/path/to/test" "/path/to/test_real" 10
restore_test_folder() {
    local test_path="$1"
    local backup_path="$2"
    local expected_count="$3"

    local n_files_sanity_check
    n_files_sanity_check=$(ls -1 "${test_path}" | wc -l)

    if [ "${n_files_sanity_check}" -ne "${expected_count}" ]; then
        # Avoid removing the test folder if it does not contain exactly 10 files
        # ⚠️ This ensures that the real test folder is not accidentally deleted
        printf "%s\n" ":::: ❌ Error: Temporary test folder ${test_path} does not contain exactly ${expected_count} files. Found: ${n_files_sanity_check} files."
        exit 1
    else
        printf "%s\n" ":::: ✅ Temporary test folder ${test_path} contains exactly ${expected_count} files. Proceeding to restore original test folder."
        # Remove temporary test folder: data/atlantic/samples/test
        rm -rf "${test_path}"
        # Rename: data/atlantic/samples/test_real -> data/atlantic/samples/test
        mv "${backup_path}" "${test_path}"
        printf "%s\n" ":::: ✅ Restored original test folder: ${test_path}"
    fi
}