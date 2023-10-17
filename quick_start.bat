@echo off
echo Cloning GeoStackingPredictor repository...
git clone https://github.com/Lukacut/GeoStackingPredictor.git
cd GeoStackingPredictor

echo Installing required packages...
pip install -r requirements.txt

echo Running main.py...
python main.py

echo Done!
pause
