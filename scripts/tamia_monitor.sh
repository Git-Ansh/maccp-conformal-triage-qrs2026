#!/bin/bash
# Monitor a TamIA SLURM job. Polls until complete, then shows status + tail of logs.
# Usage: ./scripts/tamia_monitor.sh [JOB_ID]
# If no JOB_ID given, monitors the most recent job for user anshshah.

PROJ_DIR='~/links/projects/aip-rnishat/shared/perf-regression-ci'
POLL_INTERVAL=30  # seconds between checks

JOB_ID="$1"

# If no job ID, find the most recent one
if [ -z "$JOB_ID" ]; then
    JOB_ID=$(ssh tamia "squeue -u anshshah --noheader --format='%i' | tail -1" 2>/dev/null)
    if [ -z "$JOB_ID" ]; then
        echo "No running jobs found. Checking most recent completed job..."
        # Find the most recent .out file
        JOB_ID=$(ssh tamia "ls -t ${PROJ_DIR}/jobs/*.out 2>/dev/null | head -1 | grep -oP '\d+(?=\.out)'")
        if [ -z "$JOB_ID" ]; then
            echo "No jobs found at all."
            exit 1
        fi
        echo "Most recent job: $JOB_ID (already completed)"
    fi
fi

echo "=== Monitoring TamIA job $JOB_ID ==="
echo "Poll interval: ${POLL_INTERVAL}s"
echo ""

while true; do
    # Check job state
    STATE=$(ssh tamia "sacct -j $JOB_ID --noheader --format=State%20 -P | head -1" 2>/dev/null | tr -d '[:space:]')

    if [ -z "$STATE" ]; then
        # sacct might not have it yet, try squeue
        STATE=$(ssh tamia "squeue -j $JOB_ID --noheader --format='%T'" 2>/dev/null | tr -d '[:space:]')
    fi

    TIMESTAMP=$(date '+%H:%M:%S')

    case "$STATE" in
        RUNNING)
            # Show latest output line
            LAST_LINE=$(ssh tamia "tail -1 ${PROJ_DIR}/jobs/*_${JOB_ID}.out 2>/dev/null" 2>/dev/null)
            echo "[$TIMESTAMP] RUNNING | $LAST_LINE"
            ;;
        PENDING)
            echo "[$TIMESTAMP] PENDING (waiting for resources)"
            ;;
        COMPLETED)
            echo ""
            echo "[$TIMESTAMP] === JOB COMPLETED ==="
            echo ""
            echo "--- STDOUT (last 40 lines) ---"
            ssh tamia "tail -40 ${PROJ_DIR}/jobs/*_${JOB_ID}.out" 2>/dev/null
            echo ""
            # Check if there were errors
            ERR_LINES=$(ssh tamia "grep -ciE 'error|traceback|failed|exception' ${PROJ_DIR}/jobs/*_${JOB_ID}.err 2>/dev/null" 2>/dev/null)
            if [ "$ERR_LINES" -gt 0 ] 2>/dev/null; then
                echo "--- STDERR (errors found: $ERR_LINES lines) ---"
                ssh tamia "grep -iE 'error|traceback|failed|exception|RuntimeError' ${PROJ_DIR}/jobs/*_${JOB_ID}.err 2>/dev/null" 2>/dev/null | tail -20
            else
                echo "--- No errors in stderr ---"
            fi
            exit 0
            ;;
        FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)
            echo ""
            echo "[$TIMESTAMP] === JOB $STATE ==="
            echo ""
            echo "--- STDOUT (last 20 lines) ---"
            ssh tamia "tail -20 ${PROJ_DIR}/jobs/*_${JOB_ID}.out" 2>/dev/null
            echo ""
            echo "--- STDERR (last 40 lines) ---"
            ssh tamia "tail -40 ${PROJ_DIR}/jobs/*_${JOB_ID}.err" 2>/dev/null
            exit 1
            ;;
        "")
            echo "[$TIMESTAMP] Job $JOB_ID not found in queue or accounting (may still be starting)"
            ;;
        *)
            echo "[$TIMESTAMP] State: $STATE"
            ;;
    esac

    sleep $POLL_INTERVAL
done
