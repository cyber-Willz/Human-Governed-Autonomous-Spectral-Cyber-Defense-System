# syntax=docker/dockerfile:1

# ---- Build stage -----------------------------------------------------
FROM rust:1.79-bookworm AS build
WORKDIR /src

# Cache dependency compilation separately from source changes.
COPY Cargo.toml Cargo.lock ./
COPY crates crates
RUN cargo build --release --workspace

# ---- Runtime stage -----------------------------------------------------
# A minimal debian base (not distroless/scratch) is used deliberately: the
# scanner needs a real libc for mmap/signal handling, and this keeps a shell
# available for operational debugging of what is, after all, a security
# tool that will be handed untrusted files.
FROM debian:bookworm-slim
RUN groupadd --system rclam && useradd --system --gid rclam --no-create-home rclam

COPY --from=build /src/target/release/rclamd /usr/local/bin/rclamd
COPY signatures /etc/rclam/signatures

# The daemon drops no privileges itself -- run it as an unprivileged user
# via the container/orchestrator instead of relying on in-process setuid
# logic, which is one more thing that can be gotten subtly wrong.
USER rclam

EXPOSE 3310 9310
VOLUME ["/scan"]

ENTRYPOINT ["/usr/local/bin/rclamd"]
# --allow-root /scan matches the VOLUME above: out of the box, the only
# path a client can ask this container to scan is whatever the operator
# mounted at /scan. Override --allow-root (and remount accordingly) if you
# need a different layout -- the daemon refuses every SCAN/CONTSCAN if
# --allow-root is left off entirely, so this is required, not cosmetic.
# --metrics-addr 0.0.0.0 makes /healthz and /metrics reachable from outside
# the container's network namespace, since 127.0.0.1 (the binary's own
# default) wouldn't be; bind-mount or firewall port 9310 down if it
# shouldn't be exposed as broadly as the scan port.
CMD ["--ndb", "/etc/rclam/signatures/sample.ndb", \
     "--hdb", "/etc/rclam/signatures/sample.hdb", \
     "--tcp", "--port", "3310", \
     "--allow-root", "/scan", \
     "--metrics-addr", "0.0.0.0", "--metrics-port", "9310"]
