#!/bin/bash
# Check SLURM job queue on TamIA
# Usage: ./scripts/tamia_status.sh
ssh tamia "squeue -u anshshah --format='%.10i %.30j %.8T %.10M %.6D %R'"
