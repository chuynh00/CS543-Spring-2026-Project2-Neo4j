#!/usr/bin/env bash

is_valid_java_home() {
  local candidate="${1:-}"
  [[ -n "${candidate}" && -x "${candidate}/bin/java" ]]
}

resolve_java_home() {
  if is_valid_java_home "${JAVA_HOME:-}"; then
    printf '%s\n' "${JAVA_HOME}"
    return 0
  fi

  if command -v /usr/libexec/java_home >/dev/null 2>&1; then
    local candidate=""
    for version in 21 17 ""; do
      if [[ -n "${version}" ]]; then
        candidate="$(/usr/libexec/java_home -v "${version}" 2>/dev/null || true)"
      else
        candidate="$(/usr/libexec/java_home 2>/dev/null || true)"
      fi
      if is_valid_java_home "${candidate}"; then
        printf '%s\n' "${candidate}"
        return 0
      fi
    done
  fi

  local candidate=""
  for candidate in \
    "/opt/homebrew/Cellar/openjdk/21.0.7/libexec/openjdk.jdk/Contents/Home" \
    "/opt/homebrew/Cellar/openjdk@21/21.0.7/libexec/openjdk.jdk/Contents/Home" \
    "/opt/homebrew/Cellar/openjdk@17/17.0.18/libexec/openjdk.jdk/Contents/Home" \
    "/opt/homebrew/Cellar/openjdk@17/17.0.15/libexec/openjdk.jdk/Contents/Home" \
    "/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home" \
    "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home" \
    "/usr/local/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home" \
    "/usr/local/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home" \
    "/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home" \
    "/usr/local/opt/openjdk/libexec/openjdk.jdk/Contents/Home" \
    "/opt/homebrew/Cellar/openjdk/25.0.2/libexec/openjdk.jdk/Contents/Home"
  do
    if is_valid_java_home "${candidate}"; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  if command -v brew >/dev/null 2>&1; then
    local brew_prefix
    for formula in openjdk@21 openjdk@17 openjdk; do
      brew_prefix="$(brew --prefix "${formula}" 2>/dev/null || true)"
      candidate="${brew_prefix}/libexec/openjdk.jdk/Contents/Home"
      if is_valid_java_home "${candidate}"; then
        printf '%s\n' "${candidate}"
        return 0
      fi
    done
  fi

  if command -v mvn >/dev/null 2>&1; then
    local runtime_home
    runtime_home="$(mvn -v 2>/dev/null | sed -n 's/^.*runtime: \(.*\)$/\1/p' | head -n 1)"
    if is_valid_java_home "${runtime_home}"; then
      printf '%s\n' "${runtime_home}"
      return 0
    fi
  fi

  return 1
}

ensure_java_env() {
  local resolved_home
  resolved_home="$(resolve_java_home || true)"
  if [[ -z "${resolved_home}" ]]; then
    echo "Unable to locate a usable JDK. Install Java 17 or 21 and try again."
    exit 1
  fi

  export JAVA_HOME="${resolved_home}"
  case ":${PATH}:" in
    *":${JAVA_HOME}/bin:"*) ;;
    *) export PATH="${JAVA_HOME}/bin:${PATH}" ;;
  esac
}
