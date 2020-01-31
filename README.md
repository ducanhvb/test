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
