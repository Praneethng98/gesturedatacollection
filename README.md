git clone https://github.com/yourusername/yourrepo.git
cd yourrepo
python -m venv venv             # optional, but good practice
source venv/bin/activate        # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python collect_gesture_data.py  # or whatever your script name is
