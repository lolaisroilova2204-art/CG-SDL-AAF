@echo off
REM ==== Maqola uchun to'liq benchmark (LPIPS + Blender) / Full benchmark for the manuscript ====
REM 1) pip install -e .[lpips]      2) Blender 3.6+ o'rnatilgan bo'lsin / Blender installed
REM Blender topilmasa / if not found:  set CGAAF_BLENDER=C:\Program Files\Blender Foundation\Blender 4.2\blender.exe
python -c "import lpips; lpips.LPIPS(net='alex')"
cgaaf benchmark benchmark_full --repeats 3
cgaaf paper --benchmark benchmark_full --out paper_tech
echo.
echo Natijalar / results:  paper_tech\paper_results_en.md , paper_tech\paper_tables.xlsx , paper_tech\figures\
pause
