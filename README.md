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
Mel:
```bash
tar -xf mel.tar.gz
```
Alignment:
```bash
wget https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip
unzip alignments.zip
```
Pitch:
```bash
python pitch_contour.py
```
Energy:
```bash
python energy.py
```

## FastSpeech to use waveglow code, data and audio preprocessing from this repo
```bash
gdown https://drive.google.com/u/0/uc?id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx
mkdir -p tts/waveglow/pretrained_model
mv waveglow_256channels_ljs_v2.pt tts/waveglow/pretrained/waveglow_256channels.pt
```