# Chạy thử training trên VM trước khi chuyển sang máy công ty

Mục tiêu: kiểm tra dữ liệu thật, forward/backward, gradient, cập nhật trọng số,
lưu/nạp checkpoint. Một lần chạy đạt không chứng minh model hội tụ hoặc đạt mAPH
công bố. Chưa có lần chạy CUDA nào được xác nhận trên bản chuẩn bị này.

## 1. Thu thập cấu hình máy

Tại thư mục gốc repo trên VM:

```bash
python scripts/smoke_train.py --preflight
```

Lưu kết quả GPU, driver, `nvcc`, Python, PyTorch và các dependency. Chọn bộ
dependency sau khi biết GPU và image của VM; không cài nguyên bộ CUDA/PyTorch
cũ trong INSTALL.md lên mọi loại GPU. Script này không cài phần mềm hay thuê VM.
Các CUDA extensions trong `utils/setup.py` phải build được trong môi trường đó.

## 2. Dùng tập nhỏ, tách biệt

- Bắt đầu vài sequence train có nhãn và ít nhất một sequence validation riêng.
- Dùng thư mục riêng như `/mnt/detzero-smoke/waymo`, có `ImageSets/train.txt`,
  `val.txt`, `test.txt` tương ứng dữ liệu đã chọn. Không ghi đè danh sách full.
- Preprocessor hiện dùng `data/waymo` tại root repo; trên bản checkout dành riêng
  cho VM, bố trí đường dẫn/symlink tới thư mục subset trước khi chạy preprocessing.
- Giữ toàn bộ frame của mỗi sequence; không cắt rời frame làm sai chỉ số sweep.
- Chỉ tải các TFRecord trong danh sách subset từ bucket đã xác nhận; script
  không tự tải toàn bộ bucket. TFRecord train phải chứa nhãn LiDAR 3D.
- Nếu metadata có đường dẫn tuyệt đối, chuyển máy phải giữ đường dẫn mount hoặc
  tạo lại/chuyển metadata để `lidar_path` trỏ đến file có thật.

Preprocessing (từ `detection/`, sau khi cài các package của repo):

```bash
python -m detzero_det.datasets.waymo.waymo_preprocess --cfg_file tools/cfgs/det_dataset_cfgs/waymo_1sweep.yaml --func create_waymo_infos
python -m detzero_det.datasets.waymo.waymo_preprocess --cfg_file tools/cfgs/det_dataset_cfgs/waymo_1sweep.yaml --func create_waymo_database
```

## 3. Chọn nguồn detector

- Có checkpoint CenterPoint gốc: chạy inference bằng đúng repo/config gốc, rồi
  chuyển prediction sang schema DetZero. Adapter chưa được cung cấp vì chưa có
  checkpoint/config/mẫu prediction để xác nhận class mapping và tọa độ.
- Muốn kiểm tra training detector DetZero: chạy lệnh bên dưới. Checkpoint sinh
  từ vài bước chỉ kiểm tra phần mềm; không nên dùng detector gần như ngẫu nhiên
  đó để tạo dữ liệu train refinement có ý nghĩa.
- PDV tùy chọn: dùng config PDV cùng checkpoint một giai đoạn tương thích qua
  `--pretrained-model`. Không đưa checkpoint CenterPoint gốc thẳng vào loader này.

Từ root repo, ví dụ:

```bash
python scripts/smoke_train.py --module detection --cfg_file cfgs/det_model_cfgs/centerpoint_1sweep.yaml --data-root /mnt/detzero-smoke/waymo --output /mnt/detzero-smoke/runs/det_01 --steps 10 --batch-size 2 --workers 0
```

## 4. Chuẩn bị dữ liệu refinement

1. Chạy detector có chất lượng dùng được trên subset train và validation.
2. Chạy `tracking/tools/run_track.py` cho từng split, dùng đúng config DATA_PATH.
3. Chạy `daemon/prepare_object_data.py` cho từng split và lớp; xem `--help`.
4. Kiểm tra box phủ đúng điểm LiDAR và pose/frame/sequence khớp nhau.
5. Bắt đầu với Vehicle. CRM cần cả track khớp ground truth và track không khớp;
   nếu subset không có đủ, thêm sequence thay vì tự bịa track/nhãn.

## 5. Chạy GRM và PRM riêng

Các lệnh dưới dùng dataset, model, loss, optimizer thật với config gốc, giới hạn
số bước. Không tự thay số điểm hay độ dài track, vì đó sẽ là một cấu hình khác.

```bash
python scripts/smoke_train.py --module refining --cfg_file cfgs/ref_model_cfgs/vehicle_grm_model.yaml --data-root /mnt/detzero-smoke/waymo --output /mnt/detzero-smoke/runs/grm_01 --steps 10 --batch-size 2 --workers 0
python scripts/smoke_train.py --module refining --cfg_file cfgs/ref_model_cfgs/vehicle_prm_model.yaml --data-root /mnt/detzero-smoke/waymo --output /mnt/detzero-smoke/runs/prm_01 --steps 10 --batch-size 2 --workers 0
```

Script yêu cầu output mới để tránh tự resume/ghi đè thí nghiệm. Mặc định lấy tối
đa 16 mẫu qua DataLoader, nhưng dataset refinement vẫn preload toàn bộ subset
trong ImageSets vào RAM. `--samples` không giới hạn lượng preload đó.

## 6. Tạo IoU rồi chạy CRM

Chạy `refining/tools/test.py` với config GRM và PRM tương ứng, checkpoint vừa
lưu, `--save_to_file`, và `--set DATA_CONFIG.DATA_SPLIT.test train
DATA_CONFIG.DATA_PATH /mnt/detzero-smoke/waymo`. Giữ false positive tracks trong
prediction để tạo đủ nhãn cho CRM.

Sau đó dùng `daemon/generate_iou_gt.py --class_name Vehicle --geo_path ...
--pos_path ...`. Script ghi IoU vào `data/waymo/refining/Vehicle_iou_train.pkl`
theo root repo; bảo đảm nó là cùng thư mục dữ liệu subset.

```bash
python scripts/smoke_train.py --module refining --cfg_file cfgs/ref_model_cfgs/vehicle_crm_model.yaml --data-root /mnt/detzero-smoke/waymo --output /mnt/detzero-smoke/runs/crm_01 --steps 10 --batch-size 2 --workers 0
```

GRM/PRM mới chạy vài bước có thể tạo toàn IoU thấp; khi đó lỗi thiếu gradient ở
nhánh IoU cũng có thể do dữ liệu/box suy biến. Cần xem box và phân bố IoU. Có thể
train GRM/PRM lâu hơn trên subset trước. Không dùng kết quả này kết luận chất
lượng CRM hoặc tự thay nhãn thành dữ liệu lý tưởng.

## 7. Điều kiện đạt và bước tiếp theo

`report.json` báo passed khi:

- Số bước yêu cầu chạy được trên dữ liệu thật; loss và gradient hữu hạn.
- Có gradient khác 0 và trọng số quan sát được thay đổi.
- Checkpoint lưu/nạp đúng toàn bộ model state và số bước; nạp optimizer và
  chạy tiếp một cập nhật. Scheduler được khởi tạo lại cho bước kiểm tra này;
  đây không phải kiểm chứng resume training cho kết quả giống hệt theo từng bit.

Script không chạy inference/validation tự động. Sau đó cần kiểm tra riêng:

- `test.py` nạp checkpoint và xuất đủ prediction, không NaN/Inf.
- Frame/object IDs của GRM, PRM, CRM khớp, `combine_output.py` ghép được.
- Evaluation trên split validation riêng hoàn tất. Loss thấp trên vài track
  không chứng minh model tổng quát tốt.
- Chạy thêm 100–200 bước hoặc một epoch nhỏ để đo VRAM, RAM, I/O và tính ổn định.
- Thử Pedestrian, Cyclist; thử PDV nếu sẽ dùng; thử DDP trên máy nhiều GPU riêng.
  Vehicle chạy một GPU không xác nhận các đường chạy còn lại.

Mang sang máy công ty: code đã sửa, config, danh sách subset, log,
`environment.json`, `pip-freeze.txt`, checkpoint và báo cáo. Build lại CUDA ops
cho môi trường đích, chạy lại smoke test rồi mới train đầy đủ. Cấu hình batch
size/LR cần được quyết định lại theo số GPU và tổng batch size thực tế.

## Kiểm tra CPU đã tách khỏi CUDA

```bash
python -m unittest discover -s tests -p 'test_training_preparation.py' -v
```

Các test này kiểm tra đường dẫn TFRecord/cache, metadata GT database khi chạy
lại, bỏ qua padding CRM và loss khi mask rỗng. Chúng dùng doubles ở ranh giới
CUDA/Waymo, không thay cho một lần training thực trên GPU.
