#!/bin/bash
# Watch TamIA jobs - check every 60s, report status changes
JOBS="$@"
if [ -z "$JOBS" ]; then
    echo "Usage: tamia_watch_jobs.sh JOB_ID1 JOB_ID2 ..."
    exit 1
fi

echo "=== Watching jobs: $JOBS ==="
echo "Checking every 60s..."

while true; do
    ALL_DONE=true
    for JOB in $JOBS; do
        STATE=$(ssh tamia "sacct -j $JOB --noheader --format=State%20 -P 2>/dev/null | head -1" 2>/dev/null | tr -d '[:space:]')
        
        case "$STATE" in
            RUNNING)
                LAST=$(ssh tamia "tail -1 ~/links/projects/aip-rnishat/shared/perf-regression-ci/model_ablation/logs/*_${JOB}.out 2>/dev/null" 2>/dev/null)
                echo "[$(date '+%H:%M')] Job $JOB: RUNNING | $LAST"
                ALL_DONE=false
                ;;
            PENDING)
                echo "[$(date '+%H:%M')] Job $JOB: PENDING"
                ALL_DONE=false
                ;;
            COMPLETED)
                ELAPSED=$(ssh tamia "sacct -j $JOB --noheader --format=Elapsed -P 2>/dev/null | head -1" 2>/dev/null | tr -d '[:space:]')
                if [[ "$ELAPSED" < "00:02:00" ]]; then
                    echo ""
                    echo "!!! JOB $JOB CRASHED (completed in $ELAPSED) !!!"
                    echo "=== ERROR LOG ==="
                    ssh tamia "tail -20 ~/links/projects/aip-rnishat/shared/perf-regression-ci/model_ablation/logs/*_${JOB}.err 2>/dev/null" 2>/dev/null
                    echo "=== END ERROR ==="
                else
                    echo "[$(date '+%H:%M')] Job $JOB: COMPLETED ($ELAPSED) - SUCCESS"
                    ssh tamia "tail -5 ~/links/projects/aip-rnishat/shared/perf-regression-ci/model_ablation/logs/*_${JOB}.out 2>/dev/null" 2>/dev/null
                fi
                ;;
            FAILED|TIMEOUT|CANCELLED*|OUT_OF_ME*)
                echo ""
                echo "!!! JOB $JOB FAILED ($STATE) !!!"
                echo "=== ERROR LOG ==="
                ssh tamia "tail -20 ~/links/projects/aip-rnishat/shared/perf-regression-ci/model_ablation/logs/*_${JOB}.err 2>/dev/null" 2>/dev/null
                echo "=== END ERROR ==="
                ;;
            "")
                echo "[$(date '+%H:%M')] Job $JOB: not found yet"
                ALL_DONE=false
                ;;
        esac
    done
    
    if $ALL_DONE; then
        echo ""
        echo "=== All jobs finished ==="
        break
    fi
    
    sleep 60
done
