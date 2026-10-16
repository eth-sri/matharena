# Shared argument parsing for harness launch scripts. Other arguments are kept
# in order so callers can retain their existing model/problem/billing options.
parse_container_runtime() {
  CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-docker}"
  CONTAINER_ARGS=()
  while (( $# > 0 )); do
    case "$1" in
      --container-runtime)
        if (( $# < 2 )); then
          echo "--container-runtime requires docker or podman" >&2
          return 2
        fi
        CONTAINER_RUNTIME="$2"
        shift 2
        ;;
      --container-runtime=*)
        CONTAINER_RUNTIME="${1#*=}"
        shift
        ;;
      --)
        shift
        CONTAINER_ARGS+=("$@")
        break
        ;;
      *) CONTAINER_ARGS+=("$1"); shift ;;
    esac
  done
  case "${CONTAINER_RUNTIME}" in
    docker|podman) ;;
    *) echo "--container-runtime must be docker or podman" >&2; return 2 ;;
  esac
}
