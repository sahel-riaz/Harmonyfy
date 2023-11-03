@echo off
set "URL=https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip"
set "DEST_DIR=%cd%"  REM Use the current directory as the destination

:: Download the zip file with progress
echo Downloading...

:: Use bitsadmin to download the file with progress tracking
bitsadmin /transfer "MaestroDownload" "%URL%" "%DEST_DIR%\maestro-v2.0.0-midi.zip"

:: Wait for the download to complete
:waitForDownload
timeout /t 2 /nobreak > nul
bitsadmin /list | findstr "MaestroDownload" > nul
if %errorlevel%==0 goto waitForDownload

:: Unzip the file in the same directory
powershell -Command "Expand-Archive -Path '%DEST_DIR%\maestro-v2.0.0-midi.zip' -DestinationPath '%DEST_DIR%'"

:: Remove the compressed folder
del /F /Q "%DEST_DIR%\maestro-v2.0.0-midi.zip"

:: Move the desired folder out and delete the intermediate folder
move "%DEST_DIR%\maestro-v2.0.0-midi\maestro-v2.0.0" "%DEST_DIR%"
rmdir /S /Q "%DEST_DIR%\maestro-v2.0.0-midi"

