#cot watchdog setup checkpoint
#run once gx10 and monitor are setup. confirms if
#environment is good
#if everything passes we are good to go
#if anything fails, output says what to fix
#execute in terminal 
#python setup_checkpoint.py
#exit code 0 is success
#non zero means something is broken
#configure the four constants below to match 
#deployment before running
import json
import os
import re
import sys
import time
import pathlib import Path

#where our nemotron inference server is running
#all expose OpenAI-compatible endpoints, point it
#to what we use
INFERENCE_ENDPOINT = "http://localhost:8080/v1/chat/completions"
MODEL_NAME = "nemotron-3-nano"
#this is where our openshell writes its audit log
#check nemoclaw docs for exact path in our install
#it is a common default
OPENSHELL_AUDIT_LOG = Path.home() / ".nemoclaw" / "audit.log"
#where openclaw agent stores its working state
OPENCLAW_STATE_DIR = Path.home() / ".openclaw" / "state"