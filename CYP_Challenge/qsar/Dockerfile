# Conventional-QSAR models for the CYP-Challenge activity (non-TDI) track.
#
# The image carries the ENVIRONMENT only -- no code, no data, no credentials.
# The repository is mounted at run time, so editing a script does not mean
# rebuilding the image. That is the whole point: the grid runs for hours on a
# VM and the scripts get tuned between runs.
#
# Build (local):
#     cd ~/dockerimages/CYP_Challenge/qsar
#     docker build -t cyp-qsar:1.0 .
#
# Ship to the VM (no registry):
#     docker save cyp-qsar:1.0 | gzip -9 > cyp-qsar-1.0.tar.gz
#     gcloud compute scp cyp-qsar-1.0.tar.gz vm-gpu:~ --zone us-central1-a
#     gcloud compute ssh vm-gpu --zone us-central1-a \
#         --command 'gunzip -c ~/cyp-qsar-1.0.tar.gz | docker load'
#
# Run (the --user flag keeps output files owned by you, not by root):
#     docker run --rm -it \
#         --user "$(id -u):$(id -g)" \
#         -v "$HOME/CYP-Challenge-Tutorial:/repo" \
#         -w /repo \
#         cyp-qsar:1.0 selftest
#
# GPU is NOT used by anything in this image -- PaDEL, LightGBM and every
# scikit-learn learner here are pure CPU. ChemProp is deliberately absent
# (no torch): its columns are read from disk as CSVs.
FROM mambaorg/micromamba:2.9-ubuntu22.04

# Environment first, so a code or entrypoint change does not invalidate the
# ~3 GB solve layer.
COPY --chown=$MAMBA_USER:$MAMBA_USER env.yml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml \
 && micromamba clean --all --yes \
 && rm /tmp/env.yml

# World-readable so the image still works under `--user $(id -u):$(id -g)`,
# where $HOME is not writable and the uid does not exist in /etc/passwd.
USER root
RUN chmod -R a+rX /opt/conda
COPY entrypoint.sh /usr/local/bin/cyp-entrypoint
RUN chmod 0755 /usr/local/bin/cyp-entrypoint
USER $MAMBA_USER

# Thread counts are deliberately NOT pinned here.
#
# An earlier version set OMP_NUM_THREADS=1 image-wide to stop GridSearchCV workers
# each spawning a BLAS pool. That is right for a job with outer parallelism and
# badly wrong for one without: LightGBM's num_threads=0 means "use the OpenMP
# default", so the whole arm ran on 1 core of 16 and nobody noticed because it
# still produced correct answers, just 16x slower.
#
# The caller sets the policy per run, because only the caller knows which level
# of parallelism it is using:
#     -e OMP_NUM_THREADS=8            inner threading (LightGBM, BLAS), no outer
#     -e OMP_NUM_THREADS=1 --n-jobs N outer parallelism (GridSearchCV, joblib)
# Pair either with --cpuset-cpus to partition the box between concurrent arms.
#
# MPLCONFIGDIR and HOME point somewhere writable because --user has no home.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/mpl \
    JOBLIB_TEMP_FOLDER=/tmp

WORKDIR /repo
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "/usr/local/bin/cyp-entrypoint"]
CMD ["selftest"]
