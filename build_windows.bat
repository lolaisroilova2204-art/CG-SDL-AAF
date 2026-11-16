@echo off
REM ==== CG-SDL-AAF: Windows .exe yig'ish / build Windows executable ====
REM Talab / requires: Python 3.10-3.12 (python.org), internet (bir marta / once)
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .[lpips]
pip install pyinstaller
REM LPIPS AlexNet og'irliklarini oldindan yuklab olish / pre-download LPIPS weights
python -c "import lpips; lpips.LPIPS(net='alex')"
pyinstaller --noconfirm --windowed --name CG-SDL-AAF ^
  --collect-all lpips --collect-data trimesh --collect-submodules skimage ^
  --add-data "cgaaf\blender_script.py;cgaaf" run_gui.py
echo.
echo Tayyor / Done:  dist\CG-SDL-AAF\CG-SDL-AAF.exe
pause
