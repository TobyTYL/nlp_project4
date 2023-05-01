# Text-Classification
[![](https://img.shields.io/badge/Python-3.6-blue.svg)](https://www.python.org/)
[![](https://img.shields.io/badge/pandas-0.21.0-brightgreen.svg)](https://pypi.python.org/pypi/pandas/0.21.0)
[![](https://img.shields.io/badge/numpy-1.13.1-brightgreen.svg)](https://pypi.python.org/pypi/numpy/1.13.1)



## Introduction
The classification of new text is achieved by training the text with existing labels.


## Load data

``` python
from TextClassification.load_data import load_data

# Single label
data = load_data('single')
x = data['evaluation']
y = [[i] for i in data['label']]

# Multiple labels
data = load_data('multiple')
x = [i['fact'] for i in data]
y = [i['accusation'] for i in data]
```

## Data Preprocess
* Used for pre-processing raw text data, including methods for word separation, transcoding, length unification, etc., encapsulated in TextClassification.py

``` python
preprocess = DataPreprocess()

# Processing text
texts_cut = preprocess.cut_texts(texts, word_len)
preprocess.train_tokenizer(texts_cut, num_words)
texts_seq = preprocess.text2seq(texts_cut, sentence_len)

# Tags
preprocess.creat_label_set(labels)
labels = preprocess.creat_labels(labels)
```

## Model Training and Prediction：TextClassification.py

* fit: Input the original text and labels to continue training on top of the existing model, or start training again without inputting the model<br>
* predict: Enter the original text<br>

``` python
from TextClassification import TextClassification

clf = TextClassification()
texts_seq, texts_labels = clf.get_preprocess(x_train, y_train, 
                                             word_len=1, 
                                             num_words=2000, 
                                             sentence_len=50)
clf.fit(texts_seq=texts_seq,
        texts_labels=texts_labels,
        output_type=data_type,
        epochs=10,
        batch_size=64,
        model=None)

# Save the entire module
with open('./%s.pkl' % data_type, 'wb') as f:
    pickle.dump(clf, f)

# Import the model
with open('./%s.pkl' % data_type, 'rb') as f:
    clf = pickle.load(f)
y_predict = clf.predict(x_test)
y_predict = [[clf.preprocess.label_set[i.argmax()]] for i in y_predict]
score = sum(y_predict == np.array(y_test)) / len(y_test)
print(score)  # 0.9288
```



