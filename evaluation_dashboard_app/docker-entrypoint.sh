#!/usr/bin/env bash
set -e
# Source ROS (same distro as image, matches host when built with --build-arg ROS_DISTRO=...)
if [[ -n "${ROS_DISTRO}" && -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
  source "/opt/ros/${ROS_DISTRO}/setup.bash"
fi

if [[ -d "/app/static/release_specs" ]]; then
  find /app/static/release_specs -xtype l -delete
fi

python3 -m backend.local_bbox_api &

exec streamlit run Overview.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true --server.enableStaticServing=true --server.maxMessageSize=500 "$@"
