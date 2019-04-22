# MusicGenre

### Before run

Before the run of code, you have to install `ffmpeg` package for working with `librosa` library:

`sudo apt install ffmpeg`

`pip install librosa`

After that you have to download GTZAN dataset:

`wget https://perso-etis.ensea.fr//sylvain.iloga/GTZAN+/download/gtzan.rar`

then put all music in folder `data` with genre. You can download [here](https://yadi.sk/d/-pjEdWDoRExo4Q)

`cp -r path/to/genre/* data/genre/*`

Also you have to download VGG16, InceptionV3, Mobilenet. You can find [here](https://yadi.sk/d/qae6eznYW9DbVg). And put in folder `net/{vgg,mobilenet,inception}`. 

### Run

`./inception.sh`

`./vgg.sh`

`./mobilenet.sh`


