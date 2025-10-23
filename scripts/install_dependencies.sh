pip install -r requirements.txt
cd verl/
pip install flash-attn --no-build-isolation
pip install -e .
cd ..
pip install "sglang[all]>=0.4.6.post5"