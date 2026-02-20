# Choose a directory on a drive with enough space
MY_TMPDIR="/home/aorko/workplace/Machine-Learning/tmp"
mkdir -p "$MY_TMPDIR"

# Export TMPDIR so pip uses it for temporary files
export TMPDIR="$MY_TMPDIR"

source ../venv/bin/activate

# Optional: also tell pip to cache wheels there
pip install -r ../requirements.txt --cache-dir "$MY_TMPDIR/pip_cache"


nohup python3 train_test.py > run_output.log 2>&1 &