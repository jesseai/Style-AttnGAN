echo "This script will take the words you enter on command line, and generate birds."
echo "It requires the 2106style Venv"

source /org/dp/attngan/Style-AttnGAN/code/2106style/bin/activate

echo $1  > ../data/birds/20210614.txt
python main.py --cfg cfg/eval_bird_style.yml  --text_encoder_type rnn --gpu 0 --frames $2 --makeWords $3 --render $4
