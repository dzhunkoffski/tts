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

## Train the model
```bash
python train.py --config tts/configs/fastspeech2.json
```

## Test the model
Download model config and weights from google drive https://drive.google.com/drive/folders/1ESHZLpWb0K5NimiToJaLfBb1gVTu42Y7?usp=drive_link. Then create the checkpoint folder and put model and config file there.
```bash
mkdir checkpoint
```
Then run the `test.py` script:
```bash
python test.py --config tts/configs/fastspeech2.json --resume checkpoint/model_best.pth
```
Audios are inside the `audio_samples` folder.


## Run in Kaggle
You can also test training pipeline in Kaggle with `example-notebook.ipynb`. You need to add the following into your kaggle project:
* https://www.kaggle.com/datasets/dzhunkoffski/tts-ljspeech-alignments/
* https://www.kaggle.com/datasets/dzhunkoffski/tts-ljspeech-audio/versions/2
* https://www.kaggle.com/datasets/dzhunkoffski/tts-ljspeech-energy/
* https://www.kaggle.com/datasets/dzhunkoffski/tts-ljspeech-mels/
* https://www.kaggle.com/datasets/dzhunkoffski/tts-ljspeech-pitch-contour/
* https://www.kaggle.com/datasets/dzhunkoffski/tts-pretrained-waveglow/