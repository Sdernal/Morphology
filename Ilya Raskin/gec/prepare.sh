pip install python-Levenshtein allennlp==0.9.0

# Репозиторий авторов статьи
git clone https://github.com/grammarly/gector.git

# Данные (CoNLL-2014 shared task test set)
wget https://www.comp.nus.edu.sg/~nlp/conll14st/conll14st-test-data.tar.gz
gunzip conll14st-test-data.tar.gz
tar -xvf conll14st-test-data.tar
rm conll14st-test-data.tar.gz conll14st-test-data.tar

# Эмбеддинги
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
gunzip cc.en.300.vec.gz
mkdir data
mv cc.en.300.vec data/

# Предобученный elmo
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json
mv elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5 data/weights.hdf5
mv elmo_2x4096_512_2048cnn_2xhighway_options.json data/options.json
