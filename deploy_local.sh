# Created By NirmalBorole
#!/usr/bin/env bash
# Local Deployment script for Linux/WSL
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "======================================================================"
echo "   TOMATO LEAF DISEASE AI DIAGNOSIS - LOCAL DEPLOYMENT"
echo "======================================================================"
echo "Local Server URL: http://localhost:8000"
echo "Starting Python web server..."

if [ -f "/home/nirma/tomato_env/bin/activate" ]; then
    source /home/nirma/tomato_env/bin/activate
fi

python3 server.py
