#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="/workspace/episodes/trial1"
TASK="Pick the white cube, place it in the tray, then pick the green cube and stack it on top of the white cube."
EPISODES="50"
ROBOT_IP="192.168.0.244"
DOF="3"

echo "[cube_stack] Starting data collection stack..."
cd "$REPO_ROOT"

# Launch collection in background so we can lock exposure after startup
./scripts/launch_data_collection.sh \
  --dataset "$DATASET_DIR" \
  --task "$TASK" \
  --episodes "$EPISODES" \
  --dof "$DOF" \
  --ip "$ROBOT_IP" &

COLLECT_PID=$!

# Give the camera node time to come up and converge before locking exposure.
# This keeps the flow in one command.
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  sleep 1
  echo "[cube_stack] Waiting for camera to settle... ${i}/15"
done

# Lock camera exposure once the image has converged.
ros2 service call /camera/lock_exposure std_srvs/srv/Trigger || true

echo ""
echo "[cube_stack] Camera exposure locked. Use the teleop controls below:"
echo "  R = toggle recording on/off"
echo "  N = save episode, start next"
echo "  Backspace = discard episode, restart same index"
echo "  Esc = finalize dataset and quit"
echo ""

wait "$COLLECT_PID"
