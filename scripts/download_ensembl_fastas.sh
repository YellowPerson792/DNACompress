#!/usr/bin/env bash

set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
DOWNLOAD_ROOT="${ENSEMBL_DOWNLOAD_ROOT:-/root/autodl-tmp/DNACompress/datasets/ensembl_raw}"
LOG_PATH="${DOWNLOAD_ROOT}/download.log"
PID_PATH="${DOWNLOAD_ROOT}/download.pid"
SUMMARY_PATH="${DOWNLOAD_ROOT}/expected_summary.tsv"
MANIFEST_PATH="${DOWNLOAD_ROOT}/expected_manifest.tsv"
INCLUDE_DIR="${DOWNLOAD_ROOT}/.rsync_files"

SPECIES=(
  "homo_sapiens"
  "mus_musculus"
  "bos_taurus"
  "danio_rerio"
  "drosophila_melanogaster"
  "caenorhabditis_elegans"
  "saccharomyces_cerevisiae"
  "arabidopsis_thaliana"
)

unset_proxy_env() {
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
}

source_url() {
  case "$1" in
    homo_sapiens) echo "rsync://ftp.ensembl.org/ensembl/pub/release-115/fasta/homo_sapiens/dna/" ;;
    mus_musculus) echo "rsync://ftp.ensembl.org/ensembl/pub/release-115/fasta/mus_musculus/dna/" ;;
    bos_taurus) echo "rsync://ftp.ensembl.org/ensembl/pub/release-115/fasta/bos_taurus/dna/" ;;
    danio_rerio) echo "rsync://ftp.ensembl.org/ensembl/pub/release-115/fasta/danio_rerio/dna/" ;;
    drosophila_melanogaster) echo "rsync://ftp.ensembl.org/ensembl/pub/release-115/fasta/drosophila_melanogaster/dna/" ;;
    caenorhabditis_elegans) echo "rsync://ftp.ensembl.org/ensembl/pub/release-115/fasta/caenorhabditis_elegans/dna/" ;;
    saccharomyces_cerevisiae) echo "rsync://ftp.ensembl.org/ensembl/pub/release-115/fasta/saccharomyces_cerevisiae/dna/" ;;
    arabidopsis_thaliana) echo "rsync://ftp.ensemblgenomes.ebi.ac.uk/all/pub/plants/release-62/fasta/arabidopsis_thaliana/dna/" ;;
    *) echo "Unknown species: $1" >&2; return 1 ;;
  esac
}

expected_count() {
  case "$1" in
    homo_sapiens) echo "26" ;;
    mus_musculus) echo "23" ;;
    bos_taurus) echo "33" ;;
    danio_rerio) echo "27" ;;
    drosophila_melanogaster) echo "9" ;;
    caenorhabditis_elegans) echo "7" ;;
    saccharomyces_cerevisiae) echo "17" ;;
    arabidopsis_thaliana) echo "7" ;;
    *) echo "Unknown species: $1" >&2; return 1 ;;
  esac
}

expected_bytes() {
  case "$1" in
    homo_sapiens) echo "881964081" ;;
    mus_musculus) echo "806418890" ;;
    bos_taurus) echo "833660604" ;;
    danio_rerio) echo "410230731" ;;
    drosophila_melanogaster) echo "43290101" ;;
    caenorhabditis_elegans) echo "30316631" ;;
    saccharomyces_cerevisiae) echo "3784201" ;;
    arabidopsis_thaliana) echo "36462703" ;;
    *) echo "Unknown species: $1" >&2; return 1 ;;
  esac
}

manifest_lines() {
  case "$1" in
    homo_sapiens)
      cat <<'EOF'
Homo_sapiens.GRCh38.dna.chromosome.1.fa.gz
Homo_sapiens.GRCh38.dna.chromosome.2.fa.gz
Homo_sapiens.GRCh38.dna.chromosome.3.fa.gz
Homo_sapiens.GRCh38.dna.chromosome.4.fa.gz
Homo_sapiens.GRCh38.dna.chromosome.5.fa.gz
Homo_sapiens.GRCh38.dna.chromosome.6.fa.gz
Homo_sapiens.GRCh38.dna.chromosome.7.fa.gz
Homo_sapiens.GRCh38.dna.chromosome.8.fa.gz
Homo_sapiens.GRCh38.dna.chromosome.9.fa.gz
Homo_sapiens.GRCh38.dna.chromosome.10.fa.gz
Homo_sapiens.GRCh38.dna.chromosome.11.fa.gz
Homo_sapiens.GRCh38.dna.chromosome.12.fa.gz
Homo_sapiens.GRCh38.dna.chromosome.13.fa.gz
Homo_sapiens.GRCh38.dna.chromosome.14.fa.gz
Homo_sapiens.GRCh38.dna.chromosome.15.fa.gz
Homo_sapiens.GRCh38.dna.chromosome.16.fa.gz
Homo_sapiens.GRCh38.dna.chromosome.17.fa.gz
Homo_sapiens.GRCh38.dna.chromosome.18.fa.gz
Homo_sapiens.GRCh38.dna.chromosome.19.fa.gz
Homo_sapiens.GRCh38.dna.chromosome.20.fa.gz
Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz
Homo_sapiens.GRCh38.dna.chromosome.22.fa.gz
Homo_sapiens.GRCh38.dna.chromosome.X.fa.gz
Homo_sapiens.GRCh38.dna.chromosome.Y.fa.gz
Homo_sapiens.GRCh38.dna.chromosome.MT.fa.gz
Homo_sapiens.GRCh38.dna.nonchromosomal.fa.gz
EOF
      ;;
    mus_musculus)
      cat <<'EOF'
Mus_musculus.GRCm39.dna.chromosome.1.fa.gz
Mus_musculus.GRCm39.dna.chromosome.2.fa.gz
Mus_musculus.GRCm39.dna.chromosome.3.fa.gz
Mus_musculus.GRCm39.dna.chromosome.4.fa.gz
Mus_musculus.GRCm39.dna.chromosome.5.fa.gz
Mus_musculus.GRCm39.dna.chromosome.6.fa.gz
Mus_musculus.GRCm39.dna.chromosome.7.fa.gz
Mus_musculus.GRCm39.dna.chromosome.8.fa.gz
Mus_musculus.GRCm39.dna.chromosome.9.fa.gz
Mus_musculus.GRCm39.dna.chromosome.10.fa.gz
Mus_musculus.GRCm39.dna.chromosome.11.fa.gz
Mus_musculus.GRCm39.dna.chromosome.12.fa.gz
Mus_musculus.GRCm39.dna.chromosome.13.fa.gz
Mus_musculus.GRCm39.dna.chromosome.14.fa.gz
Mus_musculus.GRCm39.dna.chromosome.15.fa.gz
Mus_musculus.GRCm39.dna.chromosome.16.fa.gz
Mus_musculus.GRCm39.dna.chromosome.17.fa.gz
Mus_musculus.GRCm39.dna.chromosome.18.fa.gz
Mus_musculus.GRCm39.dna.chromosome.19.fa.gz
Mus_musculus.GRCm39.dna.chromosome.X.fa.gz
Mus_musculus.GRCm39.dna.chromosome.Y.fa.gz
Mus_musculus.GRCm39.dna.chromosome.MT.fa.gz
Mus_musculus.GRCm39.dna.nonchromosomal.fa.gz
EOF
      ;;
    bos_taurus)
      cat <<'EOF'
Bos_taurus.ARS-UCD2.0.dna.primary_assembly.1.fa.gz
Bos_taurus.ARS-UCD2.0.dna.primary_assembly.2.fa.gz
Bos_taurus.ARS-UCD2.0.dna.primary_assembly.3.fa.gz
Bos_taurus.ARS-UCD2.0.dna.primary_assembly.4.fa.gz
Bos_taurus.ARS-UCD2.0.dna.primary_assembly.5.fa.gz
Bos_taurus.ARS-UCD2.0.dna.primary_assembly.6.fa.gz
Bos_taurus.ARS-UCD2.0.dna.primary_assembly.7.fa.gz
Bos_taurus.ARS-UCD2.0.dna.primary_assembly.8.fa.gz
Bos_taurus.ARS-UCD2.0.dna.primary_assembly.9.fa.gz
Bos_taurus.ARS-UCD2.0.dna.primary_assembly.10.fa.gz
Bos_taurus.ARS-UCD2.0.dna.primary_assembly.11.fa.gz
Bos_taurus.ARS-UCD2.0.dna.primary_assembly.12.fa.gz
Bos_taurus.ARS-UCD2.0.dna.primary_assembly.13.fa.gz
Bos_taurus.ARS-UCD2.0.dna.primary_assembly.14.fa.gz
Bos_taurus.ARS-UCD2.0.dna.primary_assembly.15.fa.gz
Bos_taurus.ARS-UCD2.0.dna.primary_assembly.16.fa.gz
Bos_taurus.ARS-UCD2.0.dna.primary_assembly.17.fa.gz
Bos_taurus.ARS-UCD2.0.dna.primary_assembly.18.fa.gz
Bos_taurus.ARS-UCD2.0.dna.primary_assembly.19.fa.gz
Bos_taurus.ARS-UCD2.0.dna.primary_assembly.20.fa.gz
Bos_taurus.ARS-UCD2.0.dna.primary_assembly.21.fa.gz
Bos_taurus.ARS-UCD2.0.dna.primary_assembly.22.fa.gz
Bos_taurus.ARS-UCD2.0.dna.primary_assembly.23.fa.gz
Bos_taurus.ARS-UCD2.0.dna.primary_assembly.24.fa.gz
Bos_taurus.ARS-UCD2.0.dna.primary_assembly.25.fa.gz
Bos_taurus.ARS-UCD2.0.dna.primary_assembly.26.fa.gz
Bos_taurus.ARS-UCD2.0.dna.primary_assembly.27.fa.gz
Bos_taurus.ARS-UCD2.0.dna.primary_assembly.28.fa.gz
Bos_taurus.ARS-UCD2.0.dna.primary_assembly.29.fa.gz
Bos_taurus.ARS-UCD2.0.dna.primary_assembly.X.fa.gz
Bos_taurus.ARS-UCD2.0.dna.primary_assembly.Y.fa.gz
Bos_taurus.ARS-UCD2.0.dna.primary_assembly.MT.fa.gz
Bos_taurus.ARS-UCD2.0.dna.nonchromosomal.fa.gz
EOF
      ;;
    danio_rerio)
      cat <<'EOF'
Danio_rerio.GRCz11.dna.chromosome.1.fa.gz
Danio_rerio.GRCz11.dna.chromosome.2.fa.gz
Danio_rerio.GRCz11.dna.chromosome.3.fa.gz
Danio_rerio.GRCz11.dna.chromosome.4.fa.gz
Danio_rerio.GRCz11.dna.chromosome.5.fa.gz
Danio_rerio.GRCz11.dna.chromosome.6.fa.gz
Danio_rerio.GRCz11.dna.chromosome.7.fa.gz
Danio_rerio.GRCz11.dna.chromosome.8.fa.gz
Danio_rerio.GRCz11.dna.chromosome.9.fa.gz
Danio_rerio.GRCz11.dna.chromosome.10.fa.gz
Danio_rerio.GRCz11.dna.chromosome.11.fa.gz
Danio_rerio.GRCz11.dna.chromosome.12.fa.gz
Danio_rerio.GRCz11.dna.chromosome.13.fa.gz
Danio_rerio.GRCz11.dna.chromosome.14.fa.gz
Danio_rerio.GRCz11.dna.chromosome.15.fa.gz
Danio_rerio.GRCz11.dna.chromosome.16.fa.gz
Danio_rerio.GRCz11.dna.chromosome.17.fa.gz
Danio_rerio.GRCz11.dna.chromosome.18.fa.gz
Danio_rerio.GRCz11.dna.chromosome.19.fa.gz
Danio_rerio.GRCz11.dna.chromosome.20.fa.gz
Danio_rerio.GRCz11.dna.chromosome.21.fa.gz
Danio_rerio.GRCz11.dna.chromosome.22.fa.gz
Danio_rerio.GRCz11.dna.chromosome.23.fa.gz
Danio_rerio.GRCz11.dna.chromosome.24.fa.gz
Danio_rerio.GRCz11.dna.chromosome.25.fa.gz
Danio_rerio.GRCz11.dna.chromosome.MT.fa.gz
Danio_rerio.GRCz11.dna.nonchromosomal.fa.gz
EOF
      ;;
    drosophila_melanogaster)
      cat <<'EOF'
Drosophila_melanogaster.BDGP6.54.dna.primary_assembly.2L.fa.gz
Drosophila_melanogaster.BDGP6.54.dna.primary_assembly.2R.fa.gz
Drosophila_melanogaster.BDGP6.54.dna.primary_assembly.3L.fa.gz
Drosophila_melanogaster.BDGP6.54.dna.primary_assembly.3R.fa.gz
Drosophila_melanogaster.BDGP6.54.dna.primary_assembly.4.fa.gz
Drosophila_melanogaster.BDGP6.54.dna.primary_assembly.X.fa.gz
Drosophila_melanogaster.BDGP6.54.dna.primary_assembly.Y.fa.gz
Drosophila_melanogaster.BDGP6.54.dna.primary_assembly.mitochondrion_genome.fa.gz
Drosophila_melanogaster.BDGP6.54.dna.nonchromosomal.fa.gz
EOF
      ;;
    caenorhabditis_elegans)
      cat <<'EOF'
Caenorhabditis_elegans.WBcel235.dna.chromosome.I.fa.gz
Caenorhabditis_elegans.WBcel235.dna.chromosome.II.fa.gz
Caenorhabditis_elegans.WBcel235.dna.chromosome.III.fa.gz
Caenorhabditis_elegans.WBcel235.dna.chromosome.IV.fa.gz
Caenorhabditis_elegans.WBcel235.dna.chromosome.V.fa.gz
Caenorhabditis_elegans.WBcel235.dna.chromosome.X.fa.gz
Caenorhabditis_elegans.WBcel235.dna.chromosome.MtDNA.fa.gz
EOF
      ;;
    saccharomyces_cerevisiae)
      cat <<'EOF'
Saccharomyces_cerevisiae.R64-1-1.dna.chromosome.I.fa.gz
Saccharomyces_cerevisiae.R64-1-1.dna.chromosome.II.fa.gz
Saccharomyces_cerevisiae.R64-1-1.dna.chromosome.III.fa.gz
Saccharomyces_cerevisiae.R64-1-1.dna.chromosome.IV.fa.gz
Saccharomyces_cerevisiae.R64-1-1.dna.chromosome.V.fa.gz
Saccharomyces_cerevisiae.R64-1-1.dna.chromosome.VI.fa.gz
Saccharomyces_cerevisiae.R64-1-1.dna.chromosome.VII.fa.gz
Saccharomyces_cerevisiae.R64-1-1.dna.chromosome.VIII.fa.gz
Saccharomyces_cerevisiae.R64-1-1.dna.chromosome.IX.fa.gz
Saccharomyces_cerevisiae.R64-1-1.dna.chromosome.X.fa.gz
Saccharomyces_cerevisiae.R64-1-1.dna.chromosome.XI.fa.gz
Saccharomyces_cerevisiae.R64-1-1.dna.chromosome.XII.fa.gz
Saccharomyces_cerevisiae.R64-1-1.dna.chromosome.XIII.fa.gz
Saccharomyces_cerevisiae.R64-1-1.dna.chromosome.XIV.fa.gz
Saccharomyces_cerevisiae.R64-1-1.dna.chromosome.XV.fa.gz
Saccharomyces_cerevisiae.R64-1-1.dna.chromosome.XVI.fa.gz
Saccharomyces_cerevisiae.R64-1-1.dna.chromosome.Mito.fa.gz
EOF
      ;;
    arabidopsis_thaliana)
      cat <<'EOF'
Arabidopsis_thaliana.TAIR10.dna.chromosome.1.fa.gz
Arabidopsis_thaliana.TAIR10.dna.chromosome.2.fa.gz
Arabidopsis_thaliana.TAIR10.dna.chromosome.3.fa.gz
Arabidopsis_thaliana.TAIR10.dna.chromosome.4.fa.gz
Arabidopsis_thaliana.TAIR10.dna.chromosome.5.fa.gz
Arabidopsis_thaliana.TAIR10.dna.chromosome.Mt.fa.gz
Arabidopsis_thaliana.TAIR10.dna.chromosome.Pt.fa.gz
EOF
      ;;
    *)
      echo "Unknown species: $1" >&2
      return 1
      ;;
  esac
}

write_include_files() {
  mkdir -p "${INCLUDE_DIR}"
  local species
  for species in "${SPECIES[@]}"; do
    manifest_lines "${species}" > "${INCLUDE_DIR}/${species}.txt"
  done
}

write_metadata() {
  mkdir -p "${DOWNLOAD_ROOT}"
  write_include_files

  {
    printf "species\texpected_count\texpected_bytes\tsource_url\n"
    local species
    for species in "${SPECIES[@]}"; do
      printf "%s\t%s\t%s\t%s\n" \
        "${species}" \
        "$(expected_count "${species}")" \
        "$(expected_bytes "${species}")" \
        "$(source_url "${species}")"
    done
  } > "${SUMMARY_PATH}"

  {
    printf "species\tfilename\n"
    local species
    while IFS= read -r species; do
      while IFS= read -r filename; do
        printf "%s\t%s\n" "${species}" "${filename}"
      done < "${INCLUDE_DIR}/${species}.txt"
    done < <(printf "%s\n" "${SPECIES[@]}")
  } > "${MANIFEST_PATH}"
}

rsync_species() {
  local species="$1"
  local mode="$2"
  local destination="${DOWNLOAD_ROOT}/${species}/dna"
  local include_file="${INCLUDE_DIR}/${species}.txt"
  local -a opts=(-avP --info=progress2 --human-readable)

  if [[ "${mode}" == "dry-run" ]]; then
    opts+=(--dry-run --itemize-changes)
  fi

  mkdir -p "${destination}"
  echo "=== ${species} (${mode}) ==="
  rsync "${opts[@]}" --files-from="${include_file}" "$(source_url "${species}")" "${destination}/"
}

run_all() {
  local mode="$1"
  unset_proxy_env
  write_metadata

  local species
  for species in "${SPECIES[@]}"; do
    rsync_species "${species}" "${mode}"
  done
}

running_pid() {
  if [[ -f "${PID_PATH}" ]]; then
    local pid
    pid="$(cat "${PID_PATH}")"
    if [[ -n "${pid}" ]] && ps -p "${pid}" > /dev/null 2>&1; then
      echo "${pid}"
      return 0
    fi
  fi
  return 1
}

start_background() {
  mkdir -p "${DOWNLOAD_ROOT}"

  local existing_pid
  if existing_pid="$(running_pid)"; then
    echo "Download already running with PID ${existing_pid}" >&2
    return 1
  fi

  local quoted_script
  quoted_script="$(printf '%q' "${SCRIPT_PATH}")"
  nohup setsid bash -lc "exec ${quoted_script} run" </dev/null > "${LOG_PATH}" 2>&1 &
  local pid=$!
  printf "%s\n" "${pid}" > "${PID_PATH}"
  echo "Started background download with PID ${pid}"
  echo "Log: ${LOG_PATH}"
}

status_download() {
  if running_pid > /dev/null; then
    local pid
    pid="$(cat "${PID_PATH}")"
    echo "running ${pid}"
    ps -fp "${pid}"
  else
    echo "not running"
    return 1
  fi
}

verify_species() {
  local species="$1"
  local dir="${DOWNLOAD_ROOT}/${species}/dna"
  local count="0"
  local bytes="0"

  if [[ -d "${dir}" ]]; then
    count="$(find "${dir}" -maxdepth 1 -type f -name '*.fa.gz' | wc -l | tr -d ' ')"
    bytes="$(find "${dir}" -maxdepth 1 -type f -name '*.fa.gz' -printf '%s\n' | awk '{sum += $1} END {print sum + 0}')"
  fi

  printf "%s\t%s\t%s\t%s\t%s\n" \
    "${species}" \
    "${count}" \
    "$(expected_count "${species}")" \
    "${bytes}" \
    "$(expected_bytes "${species}")"
}

verify_download() {
  local total_count="0"
  local total_bytes="0"
  local expected_total_count="0"
  local expected_total_bytes="0"

  printf "species\tactual_count\texpected_count\tactual_bytes\texpected_bytes\n"
  local species
  for species in "${SPECIES[@]}"; do
    local line
    line="$(verify_species "${species}")"
    echo "${line}"
    total_count="$((total_count + $(echo "${line}" | awk -F '\t' '{print $2}')))"
    total_bytes="$((total_bytes + $(echo "${line}" | awk -F '\t' '{print $4}')))"
    expected_total_count="$((expected_total_count + $(expected_count "${species}")))"
    expected_total_bytes="$((expected_total_bytes + $(expected_bytes "${species}")))"
  done

  printf "TOTAL\t%s\t%s\t%s\t%s\n" \
    "${total_count}" \
    "${expected_total_count}" \
    "${total_bytes}" \
    "${expected_total_bytes}"
}

usage() {
  cat <<EOF
Usage: $(basename "${SCRIPT_PATH}") <command>

Commands:
  dry-run  Run rsync with --dry-run and write metadata files
  run      Run the full serial download in the foreground
  start    Launch the full download in the background with nohup
  status   Show background process status from download.pid
  verify   Compare downloaded file counts and bytes against expectations
EOF
}

main() {
  local command="${1:-}"
  case "${command}" in
    dry-run) run_all "dry-run" ;;
    run) run_all "run" ;;
    start) start_background ;;
    status) status_download ;;
    verify) verify_download ;;
    *) usage; [[ -n "${command}" ]] && return 1 ;;
  esac
}

main "$@"
