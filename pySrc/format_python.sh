#!/bin/bash

# Find all .py files and run black on each
find . -type f -name "*.py" -exec black {} +

