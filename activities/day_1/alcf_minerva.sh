#!/bin/bash
# uvx alcf-tokens login inference

export OPENAI_BASE_URL="https://inference-api.alcf.anl.gov/resource_server/minerva/api/v1"
export OPENAI_API_KEY="$(uvx alcf-tokens get-token inference)"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Failed to retrieve ALCF token"
    return 1
fi

echo "ALCF Minerva configured."