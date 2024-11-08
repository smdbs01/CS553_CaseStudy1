#!/bin/bash

LOG="log.txt"
PORT=12345
DOMAIN="upward-pleasant-lobster.ngrok-free.app"

ngrok http --log $LOG --domain $DOMAIN $PORT > /dev/null &