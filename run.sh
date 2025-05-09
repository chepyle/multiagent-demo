#!/usr/bin/env bash

# run this to start the app
# This script is used to start the Autogen Studio UI with a specified app directory.

if [ "$#" -gt 1 ]; then
  echo "Error: Too many arguments provided. Only one argument is allowed."
  exit 1
fi

APP_DIR=${1:-./app} # Use the first argument as the directory or default to ./app

read -sp "Enter password for unzip: " PASSWORD
echo
# app was previously zipped with `zip -oer -P sapa app.zip app/` which encrypts the zip file with a password
unzip -oP "$PASSWORD" "${APP_DIR}.zip"

autogenstudio ui --port 8081 --appdir "$APP_DIR"
