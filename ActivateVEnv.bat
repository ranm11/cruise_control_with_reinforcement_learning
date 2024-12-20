rem install Visual Studio 9: https://www.microsoft.com/en-us/download/confirmation.aspx?id=44266

pushd %~dp0
set VENV=..\cruise_ctrl_venv
if not exist %VENV% (
    call pip install venv
    call python -m venv %VENV%
    call %VENV%\Scripts\activate.bat
    call pip install tensorflow  matplotlib gym numpy keras
) ELSE (
    call %VENV%\Scripts\activate.bat
)
popd
