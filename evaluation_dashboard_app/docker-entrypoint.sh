#!/usr/bin/env bash
set -e
# Source ROS (same distro as image, matches host when built with --build-arg ROS_DISTRO=...)
if [[ -n "${ROS_DISTRO}" && -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
  source "/opt/ros/${ROS_DISTRO}/setup.bash"
fi

exec streamlit run Overview.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true --server.enableStaticServing=true "$@"
