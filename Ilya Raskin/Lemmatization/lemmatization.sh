mkdir -p data/
wget -O data/train.conllu https://github.com/UniversalDependencies/UD_Russian-GSD/blob/master/ru_gsd-ud-train.conllu?raw=true
wget -O data/dev.conllu https://github.com/UniversalDependencies/UD_Russian-GSD/blob/master/ru_gsd-ud-dev.conllu?raw=true
wget -O data/test.conllu https://github.com/UniversalDependencies/UD_Russian-GSD/blob/master/ru_gsd-ud-test.conllu?raw=true

pip install allennlp allennlp-models conllu nltk

# train
rm -rf res_model
allennlp train -s res_model src/config.json

# test
allennlp evaluate res_model/model.tar.gz data/test.conllu