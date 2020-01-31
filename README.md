#  Giới thiệu mạng hồi quy (Recurrent Neural Networks)
## Đặt vấn đề 
Hãy mở đầu bài viết với một câu tiếng anh - "working love learning we on deep", bạn có hiểu ý nghĩa câu trên không? 
Chắc là không đâu nhỉ bởi câu đúng phải là - "We love working on deep learning". Sự thay đổi trật tự từ làm câu trở nên 
rời rạc, khó hiểu. Vậy ta có thể mong đợi mạng nơron hiểu được ý nghĩa của nó. Chắc hẳn là không, bởi nếu bộ não người
còn băn khoăn về nó thì chắc chắn một mạng nơ ron đơn thuần cũng sẽ gặp khó khi giãi mã văn bản như vậy.\
Có nhiều nhiệm vụ như vậy trong cuộc sống hàng ngày bị phá vỡ hoàn toàn khi trình tự của chúng bị xáo trộn. Chẳng hạn, 
ngôn ngữ như chúng ta đã thấy trước đó - chuỗi từ xác định nghĩa của chúng, dữ liệu chuỗi thời gian - trong đó thời gian 
xác định sự xuất hiện của sự kiện, dữ liệu của trình tự bộ gen - trong đó mỗi chuỗi có một ý nghĩa khác nhau. Có nhiều 
trường hợp như vậy trong đó chuỗi thông tin xác định chính sự kiện. Nếu chúng ta đang cố gắng sử dụng dữ liệu đó cho bất 
kỳ đầu ra hợp lý nào, chúng ta cần một mạng có khả năng truy cập vào một số thông tin trước đó của dữ liệu để hiểu hoàn
toàn dữ liệu đó. Khi đó mạng hồi quy (Recurrent Neural Networks) được đề xuất.  
## Cần một mạng nơ-ron xử lý với các chuỗi 
Trước khi đi sâu vào chi tiết mạng hồi quy là gì, hãy suy ngẫm một chút liệu chúng ta có thực sự cần một mạng đặc biệt
để xử lý các chuỗi thông tin và các nhiệm vụ mà ta có thể đạt được bằng cách sử dụng các mạng như vậy. \
Vẻ đẹp của mạng hồi quy nằm ở sự đa dạng của ứng dụng. Khi chúng ta làm việc với RNN, chúng có khả năng tuyệt vời để đối 
phó với các loại đầu vào và đầu ra khác nhau. 

- Phân loại sắc thái câu - Đây có thể là một nhiệm vụ đơn giản là phân loại các bình luận trên twitter thành tích cực hay 
tiêu cực. Vì vậy, ở đây đầu vào là các câu trên twitter có độ dài ngắn khác nhau, trong khi đầu ra là cố định.

- Đặt tiêu đề cho ảnh - Ở đây, hãy giả sử rằng chúng ta có một hình ảnh và cần đặt một tiêu đề cho nó. Vì vậy, chúng ta 
có đầu vào là một ảnh, và một từ hay một câu là đầu ra. Ở đây, hình ảnh có thể có kích thước cố định, nhưng đầu ra mô tả 
có thể có độ dài khác nhau.

- Dịch thuật - Điều này về cơ bản có nghĩa là chúng ta có một số văn bản trong một ngôn ngữ cụ thể, ví dụ tiếng anh và 
chúng ta muốn dịch nó sang tiếng Pháp. Mỗi ngôn ngữ có ngữ pháp riêng của nó và sẽ có độ dài ngắn khác nhau cho cùng một
câu. Ở đây đầu vào và đầu ra có độ dài khác nhau.

Vì vậy, RNN có thể được sử dụng để ánh xạ đầu vào thành đầu ra với các loại, độ dài khác nhau. Hãy cùng xem kiến trúc 
của một mạng hồi quy như thế nào?
## Recurrent Neural Networks là gì?
Với nhiệm vụ là dự đoán từ tiếp theo trong câu. Hãy cùng cố gắng hoàn thành nó bằng Multilayer Perceptron (MLP). Ở dạng
đơn giản nhất, chúng ta có một lớp đầu vào, một lớp ẩn và một lớp đầu ra. Lớp đầu vào nhận đầu vào, lớp kích hoạt ẩn được 
áp dụng và cuối cùng ta nhận được đầu ra.

<p align="center">
  <img src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/12/07111304/RNN-183x300.png">
</p>

Hãy xem xét mạng có nhiều lớp ẩn. Lớp đầu vào nhận đầu vào, kích hoạt lớp ẩn đầu tiên được áp dụng và sau đó các kích 
hoạt này được gửi đến lớp ẩn tiếp theo và kích hoạt liên tiếp qua các lớp để tạo đầu ra. Mỗi lớp ẩn được đặc trưng bởi 
weights và biases riêng của nó.\
Vì mỗi lớp ẩn có weights và biases riêng, chúng hoạt động độc lập. Bây giờ mục tiêu là xác định mối quan hệ giữa các đầu 
vào liên tiếp. Chúng ta có thể cung cấp đầu vào cho các lớp ẩn? Câu trả lời là có.\

<p align="center">
  <img width="350" height="400" src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/12/07120933/rnn1234561-768x802.png">
</p>

Weights và bias của các lớp ẩn này là khác nhau. Do đó, mỗi lớp này hoạt động độc lập và không thể kết hợp với nhau. Để
kết hợp các lớp ẩn này lại với nhau, chúng phải có cùng weights và bias cho các lớp ẩn này.\

<p align="center">
  <img width="350" height="400" src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/12/07121550/rnn-intermediate-768x802.png">
</p>

Bây giờ chúng ta có thể kết hợp các lớp này lại với nhau. Tất cả các lớp ẩn này có thể được cuộn lại với nhau trong một 
lớp lặp lại duy nhất. 

<p align="center">
  <img width="180" height="300" src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/12/07113713/rnn1.png">
</p>

Ở tất cả các bước thời gian, weights của mạng hồi quy sẽ giống nhau vì hiện tại nó là một nơ-ron đơn. Nơ-ron hồi quy lưu 
trữ trạng thái của đầu vào trước đó và kết hợp với đầu vào hiện tại do đó duy trì một số mối quan hệ của đầu vào hiện tại 
với đầu vào trước đó.\
## Hiểu sâu về một Recurrent Nơ-ron 
Đầu tiên, hãy xét một nhiệm vụ đơn giản. Chúng ta có từ 'hello', ta cung cấp 4 chữ cái: h, e, l, l và yêu cầu mạng dự 
đoán chữ cái tiếp theo 'o'. Từ vựng bao gồm 4 chữ cái: {h, e, l, o}. 

<p align="center">
  <img width="195" height="300" src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/12/05231650/rnn-neuron-196x300.png">
</p>

Hãy xem cách mà kiến trúc trên dự đoán chữ cái thứ 5. Trong cấu trúc trên, khối RNN áp dụng công thức hồi quy (Recurrence 
fomular) cho véc-tơ đầu vào và trạng thái phía trước. Trong trường hợp này, chữ cái 'h' không có gì phía trước nên hãy xét 
chữ 'e'. Tại thời điểm t, 'e' là đầu vào mạng, tại t-1, đầu vào là h. Công thức xác định trạng thái hiện tại:
 
<p align="center">
  <img width="100" height="40" src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/12/06004252/hidden-state.png">
</p>

Trong đó, ht là trạng thái tại thời điểm t, ht-1 là trạng thái tại t-1, xt là đầu vào tại thời điểm t. Nếu hàm 
activation là hàm tanh, weight tại recurrent nơ-ron là Whh, weight tại input nơ-ron là Wxh thì ta sẽ có hàm xác định ht là:

<p align="center">
  <img width="200" height="40" src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/12/06005300/eq2.png">
</p>

Đầu ra được tính theo công thức:

<p align="center">
  <img width="75" height="40" src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/12/06005750/outeq.png">
</p>

Hãy tóm tắt các bước tính toán cho một recurrent nơ-ron:
1. Đầu vào cấp cho mạng, xt.
2. Tính toán trạng thái hiện tại ht bằng cách sử dụng đầu vào xt và trạng thái trước đó ht-1.
3. Trạng thái hiện tại ht trở thành ht-1 ở bước tiếp theo.
4. Chúng ta có thể đi nhiều bước tùy thuộc vào yêu cầu đề bài và kết hợp thông tin từ tất cả các trạng thái trước đó.
5. Khi tất cả các bước được hoàn thành, trạng thái hiện tại được sử dụng để tính toán đầu ra yt.
6. Sau đó đầu ra yt sẽ được so sánh với đầu ra thực tế để tính toán sai số.
7. Sai số được sử dụng trong quá trình lan truyền ngược để cập nhật weights và nhờ đó mạng được đào tạo.

Hãy xem cách chúng ta tính toán các trạng thái và đầu ra mạng qua ví dụ.

## Lan truyền xuôi (Forward Propagation) trong recurrent nơ-ron trên excel
<p align="center">
  <img width="180" height="100" src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/12/06010908/inputs.png">
</p>

Bộ từ vựng: {h, e, l, o}. Đầu vào được mã hóa thành các véc-tơ one hot. Nơ-ron đầu vào sẽ được chuyển thành trạng thái ẩn 
nhờ ma trận Wxh. Ta khởi tạo ma trận 3x4 bất kỳ.
<p align="center">
  <img width="250" height="120" src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/12/06011846/wxh.png">
</p>

**step 1:**\
Đầu vào là chữ cái 'h'. Wxh*Xt được xác định:
<p align="center">
  <img width="500" height="150" src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/12/06122426/first-state-h.png">
</p>

**step 2:**\
Whh là ma trận 1x1 <img width="70" height="40" src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/12/06013320/WHH.png">
và bias cũng là ma trận 1x1 <img width="70" height="40" src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/12/06013447/bias.png">\
Với chữ 'h', trạng thái trước đó là [0,0,0]. Giá trị whh*ht-1+bias:
<p align="center">
  <img width="450" height="150" src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/12/WHHT-1-1.png">
</p>

**step 3:**\
Gía trị trạng thái hiện tại ht được tính theo công thức: <img width="300" height="50" src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/12/06014059/eq21.png">

<p align="center">
  <img width="700" height="300" src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/12/06130247/ht-h.png">
</p>

**step 4:**\
Tiếp tục với trạng thái tiếp, 'e' là đầu vào mạng. ht giờ trở thành ht-1, và xt là véc-tơ one hot của 'e'.\
Gía trị *Whhht-1 + bias là:
<p align="center">
  <img width="800" height="100" src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/12/06131259/new-ht-1.png">
</p>

Wxh*xt:
<p align="center">
  <img width="600" height="150" src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/12/06132150/state-e.png">
</p>

**step 5:**\
ht tại chữ 'e'.
<p align="center">
  <img width="600" height="150" src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/12/06132639/htletter-e.png">
</p>

**step 6:**\
Tại mỗi trạng thái, Mạng sẽ tính được giá trị đầu ra. Hãy tính yt cho chữ cái 'e'.\
<p align="center">
  <img width="100" height="50" src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/12/06005750/outeq.png">
</p>
<p align="center">
  <img width="600" height="150" src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/12/06133208/ytfinal123.png">
</p>

**step 7:**\
Xác xuất xuất hiện tại đầu ra của các chữ cái trong bộ từ vựng được xác định bằng hàm softmax.
<p align="center">
  <img width="600" height="150" src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/12/06133614/classwise-prob.png">
</p>

Nhìn vào kết quả trên, ta thấy model chỉ ra rằng sau chữ 'e' nên là chữ 'h' vì nó có xác suất cao nhất. Vậy liệu có điều 
gì không đúng không nhỉ? Không đâu, vì chúng ta chưa huấn luyện mạng. Chúng ta mới chỉ cho nó hai chữ cái mà chưa dạy nó học.\
Bây giờ câu hỏi lớn tiếp theo ta phải đối mặt là quá trình lan truyền ngược trong mạng hồi quy diễn ra như thế nào? Bằng 
cách nào các tham số weights được cập nhật?
## Lan truyền ngược trong mạng hồi quy (BPTT)
Để tưởng tượng bằng cách nào các tham số weights được cập nhật trong trường hợp mạng hồi quy có thể sẽ có đôi chút khó 
khăn. Để hiểu và minh họa quá trình lan truyền ngược, hãy trải mạng ra tại các bước thời gian. Trong mạng hồi quy, chúng ta 
có thể có hoặc không có đầu ra tại mỗi bước thời gian.\
<p align="center">
  <img width="600" height="200" src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/12/06022525/bptt.png">
</p>

Trong trường hợp mạng hồi quy, nếu yt là giá trị dự đoán, ȳt là giá trị thực, sai số được tính theo hàm cross entropy:\

Et(ȳt,yt) = – ȳt log(yt)

E(ȳ,y) = – ∑ ȳt log(yt)

Chúng ta thường coi toàn bộ một chuỗi (một từ) là một ví dụ để huấn luyện, do đó, sai số tổng là tổng của các sai số tại 
mỗi thời điểm (kí tự). Như chúng ta thấy, weights là giống nhau ở mỗi bước thời gian. Hãy tóm tắt các bước trong lan truyền 
ngược:\
1. Cross entropy error được tính dựa trên giá trị đầu ra tính toán và đầu ra thực.
2. Hãy nhớ rằng mạng được trải ra tại tất cả các bước thời gian.
3. Đối với mạng được trải ra, gradient được tính cho mỗi bước thời gian đối với các tham số weight.
4. Bây giờ tham số weight là như nhau tại tất cả các bước thời gian, gradients có thể được kết hợp với nhau cho tất cả các 
bước thời gian.
5. Các weights được cập nhất cho cả recurrent nơ-ron và các dense layers.

Mạng hồi quy trải ra trông giống như một mạng nơ-ron thông thường và thuật toán lan truyền ngược tương tự như một mạng 
nơ-ron thông thường, chỉ là chúng ta kết hợp các gradient của error cho toàn bộ các bước thời gian. Bây giờ bạn nghĩ điều 
gì có thể xảy ra nếu có 100 bước thời gian. Điều này về cơ bản sẽ mất rất nhiều thời gian để mạng hội tụ vì sau khi trải 
ra, mạng sẽ trở lên rất lớn.
## Thực hiện mạng hồi quy với Keras 
Sử dụng mạng hồi quy để dự đoán sắc thái câu trên tweets. Chúng ta sẽ dự đoán trạng thái tích cực hay tiêu cực của câu. 
Dữ liệu có thể tải tại [đây](https://github.com/crwong/cs224u-project/tree/master/data/sentiment).\
Chúng ta có khoảng 1600000 câu để huấn luyện mạng. Nếu bạn chưa quen với những khái niệm cơ bản trong NLP, bạn có thể
đọc bài viết [này](https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/)
hoặc bài viết về word embedding tại [đây](https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/)

```python
# import all libraries
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.convolutional import Conv1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import spacy
nlp=spacy.load("en")

#load the dataset
train=pd.read_csv("../datasets/training.1600000.processed.noemoticon.csv" , encoding= "latin-1")
Y_train = train[train.columns[0]]
X_train = train[train.columns[5]]

# split the data into test and train
from sklearn.model_selection import train_test_split
trainset1x, trainset2x, trainset1y, trainset2y = train_test_split(X_train.values, Y_train.values, test_size=0.02,random_state=42 )
trainset2y=pd.get_dummies(trainset2y)

# function to remove stopwords
def stopwords(sentence):
   new=[]
   sentence=nlp(sentence)
    for w in sentence:
        if (w.is_stop == False) & (w.pos_ !="PUNCT"):
            new.append(w.string.strip())
        c=" ".join(str(x) for x in new)
    return c

# function to lemmatize the tweets
def lemmatize(sentence):
    sentence=nlp(sentence)
    str=""
    for w in sentence:
        str+=" "+w.lemma_
    return nlp(str)

#loading the glove model
def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
    print ("Done."),len(model),(" words loaded!")
    return model

# save the glove model
model=loadGloveModel("/mnt/hdd/datasets/glove/glove.twitter.27B.200d.txt")

#vectorising the sentences
def sent_vectorizer(sent, model):
    sent_vec = np.zeros(200)
    numw = 0
    for w in sent.split():
        try:
            sent_vec = np.add(sent_vec, model[str(w)])
            numw+=1
        except:
            pass
    return sent_vec

#obtain a clean vector
cleanvector=[]
for i in range(trainset2x.shape[0]):
    document=trainset2x[i]
    document=document.lower()
    document=lemmatize(document)
    document=str(document)
    cleanvector.append(sent_vectorizer(document,model))

#Getting the input and output in proper shape
cleanvector=np.array(cleanvector)
cleanvector =cleanvector.reshape(len(cleanvector),200,1)

#tokenizing the sequences
tokenizer = Tokenizer(num_words=16000)
tokenizer.fit_on_texts(trainset2x)
sequences = tokenizer.texts_to_sequences(trainset2x)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=15, padding="post")
print(data.shape)

#reshape the data and preparing to train
data=data.reshape(len(cleanvector),15,1)
from sklearn.model_selection import train_test_split
trainx, validx, trainy, validy = train_test_split(data, trainset2y, test_size=0.3,random_state=42 )
```
```python 
#calculate the number of words
nb_words=len(tokenizer.word_index)+1

#obtain theembedding matrix
embedding_matrix = np.zeros((nb_words, 200))
for word, i in word_index.items():
    embedding_vector = model.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

trainy=np.array(trainy)
validy=np.array(validy)

#building a simple RNN model
def modelbuild():
    model = Sequential()
    model.add(keras.layers.InputLayer(input_shape=(15,1)))
    keras.layers.embeddings.Embedding(nb_words, 15, weights=[embedding_matrix], input_length=15,
    trainable=False)
 
    model.add(keras.layers.recurrent.SimpleRNN(units = 100, activation='relu',
    use_bias=True))
    model.add(keras.layers.Dense(units=1000, input_dim = 2000, activation='sigmoid'))
    model.add(keras.layers.Dense(units=500, input_dim=1000, activation='relu'))
    model.add(keras.layers.Dense(units=2, input_dim=500,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
   
#compiling the model
finalmodel = modelbuild()
finalmodel.fit(trainx, trainy, epochs=10, batch_size=120,validation_data=(validx,validy))
```

Sau khi chạy model trên, nó có thể không cung cấp cho bạn kết quả tốt nhất vì đây là một kiến trúc cực kỳ đơn giản và chưa 
sâu. Bạn có thể thay đổi kiến trúc mạng để có được kết quả tốt hơn. Ngoài ra , bạn có thể thử các bước tiền xử lý dữ liệu.

## Kết luận
Hy vọng bài viết sẽ giúp các bạn làm quen với mạng hồi quy. Trong các bài viết sắp tới, chúng ta sẽ đi sâu vào toán học của 
mạng hồi quy và các biến thể của mạng RNN cơ bản như: GRU, LSTM. Hãy thử khám phá các kiến trúc của mạng RNN và bạn sẽ 
ngạc nhiên bởi tính hiệu quả của chúng trong các ứng dụng. Hãy để lại góp ý của bạn trong phần bình luận để giúp chúng tôi 
cải thiện bài viết. 