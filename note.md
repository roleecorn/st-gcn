# 資料集
資料量 240436筆 train 19796 test
## data.npy
一個五維的numpy物件
```
val_data = np.load('val_data.npy')
```
可以直接打開，五維分別代表
```
19796：這是測試集中的樣本數。論文中提到，該數據集包含240,000個訓練樣本和20,000個驗證樣本，因此測試集中可能有20,000個樣本之外的其他樣本。
3：這是每個時間戳的空間卷積核大小。論文中沒有提供詳細的說明，但是這個參數可能會影響模型的學習能力和訓練時間。
300：這是每個樣本中的時間戳數量。每個樣本都是10秒的視頻片段，轉換成由18個關節點組成的骨架序列後，每秒會產生30個時間戳，因此每個樣本有300個時間戳。
18：這是每個骨架序列中的關節點數量。在論文中，每個骨架序列都由18個關節點組成。
2：這是每個關節點的坐標維度。從資料上來看應該是在
(-0.495, -0.495)~(1.325, 1.309)之間
```
## label.pkl
包含兩個部分的資料
```
label_path='./val_label.pkl'
with open(label_path, 'rb') as f:
    sample_name, label = pickle.load(f)
```
可以打開，sample_name, label都是一個list，因為長度跟val_data一樣

所以可以確定val_data的第一個位置代表著樣本數

## 繼承關係
REC_Processor繼承自Processor，而Processor繼承自IO

但是Processor的init_environment會調用IO的init_environment

此時會創造一個名為self.io的IO類別物件

REC_Processor沒有init，所以用Processor的init
```
self.load_arg(argv) ->IO
self.init_environment() ->Processor
self.load_model() ->REC_Processor
# 這裡load_model是由st_gcn.py得出的，它使用了Graph 的class
self.load_weights() ->IO
self.gpu() ->IO
self.load_data() ->Processor
self.load_optimizer() ->REC_Processor
```
## Graph
```
self.max_hop = max_hop
self.dilation = dilation

self.get_edge(layout)
self.hop_dis = get_hop_distance(
    self.num_node, self.edge, max_hop=max_hop)
self.get_adjacency(strategy)
```
### get_edge
直接使用是先定義好的關係，layout選擇定義的是哪一種關係
`self.num_node = 18`會告訴名行有多少點，因此這裡是18個點

`val_data.shape=(19796, 3, 300, 18, 2)`

19796：這是測試集中的樣本數。論文中提到，該數據集包含240,000個訓練樣本和20,000個驗證樣本，因此測試集中可能有20,000個樣本之外的其他樣本。

3：這是每個時間戳的空間卷積核大小。論文中沒有提供詳細的說明，但是這個參數可能會影響模型的學習能力和訓練時間。

300：這是每個樣本中的時間戳數量。每個樣本都是10秒的視頻片段，轉換成由18個關節點組成的骨架序列後，每秒會產生30個時間戳，因此每個樣本有300個時間戳。

18：這是每個骨架序列中的關節點數量。在論文中，每個骨架序列都由18個關節點組成。

2：這是每個關節點的坐標維度。從資料上來看應該是在
(-0.495, -0.495)~(1.325, 1.309)之間

###  link
分為三種

self_link 自連接

neighbor_1base 自定的

neighbor_link 由neighbor_1base演化的自定義

self.center = 1 需要被是先定義好

## Processor.load_data

這一段程式碼是建立一個名為train的資料載入器(DataLoader)，其作用是將數據分為一個個小批次(batch)，以便模型訓練。

具體來說，這個資料載入器的建立過程如下：
```
創建一個Feeder數據集對象，並傳入名為train_feeder_args的參數，該參數是一個字典，包含了建立Feeder對象時需要的一些參數。
設置每個小批次(batch)的大小為batch_size，batch_size是一個參數，由外部傳入。
設置是否對數據進行洗牌(shuffle)，這裡將其設置為True，表示需要洗牌。
設置使用多少個num_workers進程來加載數據，這個參數是由self.arg.num_worker和torchlight.ngpu(self.arg.device)兩個參數計算得到的，其中self.arg.num_worker表示使用的CPU進程數，torchlight.ngpu(self.arg.device)表示使用的GPU數量。
設置是否保留最後一個小批次(drop_last)，這裡將其設置為True，表示當最後一個小批次的數據量不足batch_size時，丟棄最後一個小批次。
總體而言，這個資料載入器將Feeder數據集對象中的數據切分為大小為batch_size的小批次，並對其進行洗牌(shuffle)和多進程加載，以加快數據的載入速度。
```
feeder: feeder.feeder.Feeder
```
Feeder是一個用於動作識別(action recognition)的輸入資料處理類別，其用於讀取動作識別資料集並進行前處理。通常，動作識別資料集中的資料是由人體骨架的運動軌跡數據所構成，因此Feeder類別通常用於處理這類資料。

在這個特定的程式碼中，Feeder的主要功能是從.npy格式的數據檔案中讀取骨架運動軌跡數據，對數據進行前處理，包括：隨機選擇部分輸入序列、隨機位移輸入序列、將輸入序列截斷到固定的窗口大小、對輸入序列進行歸一化等操作。最終返回一個包含輸入序列和標籤(label)的資料對象，用於訓練模型。

需要注意的是，Feeder的具體實現方式可能因應不同的應用場景而略有不同
debug模式下只保留一部分的資料使用
```
讀取資料時用的code
```
with open(self.label_path, 'rb') as f:
    self.sample_name, self.label = pickle.load(f)

# load data
if mmap:
    self.data = np.load(self.data_path, mmap_mode='r')
else:
    self.data = np.load(self.data_path)
```

提到很多次的self.io
```
def init_environment(self):
    self.io = torchlight.IO(
        self.arg.work_dir,
        save_log=self.arg.save_log,
        print_log=self.arg.print_log)
    self.io.save_arg(self.arg)
```
IO應該是
`st-gcn/torchlight/build/lib/torchlight/io.py`
的IO，在這裡會被初始化
```
def init_environment(self):
    super().init_environment()
```