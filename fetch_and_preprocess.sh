#!/bin/bash
set -e # exit on error
python3 scripts/download.py

CLASSPATH="lib:lib/stanford-parser/stanford-parser.jar:lib/stanford-parser/stanford-parser-3.5.1-models.jar"
javac -cp $CLASSPATH lib/*.java
python3 scripts/preprocess-sst.py

