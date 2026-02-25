apt update -y
apt install -y python3-pip
apt install -y wget
apt install -y python3-venv
python3 -m venv pyenv

./pyenv/bin/python -m pip install -r requirements.txt
./pyenv/bin/python -m pip install vllm[flashinfer]

openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes