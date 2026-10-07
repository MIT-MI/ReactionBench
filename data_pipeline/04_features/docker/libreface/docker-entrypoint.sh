#!/usr/bin/env bash
set -euo pipefail

if [[ ! -x "${LIBREFACE_ENV}/bin/python" ]]; then
  echo "Missing mounted conda env at ${LIBREFACE_ENV}" >&2
  echo "Mount your host libreface conda env at ${LIBREFACE_ENV} inside the container (see run_libreface.sh)." >&2
  exit 1
fi

export PATH="${LIBREFACE_ENV}/bin:${PATH}"

exec "$@"
