# tts
DL-AUDIO homework

## Data:
Audio:
```bash
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -o /dev/null --show-progress
mkdir data
tar -xvf LJSpeech-1.1.tar.bz2 >> /dev/null
mv LJSpeech-1.1 data/LJSpeech-1.1
```
Text:
```bash
gdown https://drive.google.com/u/0/uc?id=1-EdH0t0loc6vPiuVtXdhsDtzygWNSNZx
mv train.txt data/LJSpeech-1.1
```

## FastSpeech to use waveglow code, data and audio preprocessing from this repo
```bash
git clone https://github.com/xcmyz/FastSpeech.git
mv FastSpeech/text tts
mv FastSpeech/audio tts
mkdir tts/waveglow
mv FastSpeech/waveglow/* tts/waveglow/
mv FastSpeech/utils.py tts
mv tts/utils.py tts/vocode_utils.py
mv FastSpeech/glow.py tts
gdown https://drive.google.com/u/0/uc?id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx
mkdir -p tts/waveglow/pretrained_model
mv waveglow_256channels_ljs_v2.pt tts/waveglow/pretrained/waveglow_256channels.pt
```

## Download mel-spectrograms
Download archive manually and put it inside the project root folder.
```bash
tar -xf mel.tar.gz
```

## Get alignments
```bash
wget https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip
unzip alignments.zip
```