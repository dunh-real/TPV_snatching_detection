# Tracker benchmark

Thư mục này chạy nhiều tracker trên cùng model detection và cùng tập video mà không ghi vào database chính.

## Chạy nhanh

Từ thư mục gốc repository:

```powershell
.\.venv\Scripts\python.exe -m experiments.tracker_benchmark.run --list-profiles

.\.venv\Scripts\python.exe -m experiments.tracker_benchmark.run `
  --videos data/videos `
  --output data/tracker_benchmark
```

Mặc định chạy sáu profile đại diện:

- `bytetrack_tuned`
- `botsort_reid_tuned`
- `ocsort_default`
- `deepocsort_default`
- `fasttrack_default`
- `tracktrack_default`

Đây là so sánh **profile thực tế**, không phải thuật toán thuần túy: ByteTrack và BoT-SORT dùng cấu hình project đã tune, còn bốn tracker khác dùng cấu hình Ultralytics mặc định. Khi shortlist được tracker mới, hãy tune nó trên validation set riêng rồi benchmark lại.

So sánh riêng custom và default:

```powershell
.\.venv\Scripts\python.exe -m experiments.tracker_benchmark.run `
  --profiles bytetrack_tuned bytetrack_default botsort_reid_tuned botsort_default
```

Debug trên 30 frame và lưu video visualization:

```powershell
.\.venv\Scripts\python.exe -m experiments.tracker_benchmark.run `
  --max-frames 30 `
  --save-videos
```

## Output

Mỗi run tạo một thư mục timestamp trong `data/tracker_benchmark`:

- `report.md`: báo cáo tổng hợp, shortlist và ưu/nhược điểm.
- `summary.csv`: metric theo tracker/video.
- `tracks.csv`: thống kê từng track.
- `fragment_candidates.csv`: các cặp track có khả năng là cùng object bị đổi ID.
- `observations.csv`: bbox, ID, class và confidence theo frame.
- `metadata.json`: phiên bản thư viện, cấu hình, model và command.
- `errors.csv`: profile/video lỗi, nếu có.
- `videos/`: chỉ xuất hiện khi dùng `--save-videos`.

## Ý nghĩa metric

- `steady_tracker_fps`: FPS sau khi bỏ thời gian setup frame đầu.
- `short_track_ratio`: tỷ lệ track có tối đa 5 observation.
- `internal_gap_frames`: số frame mất ở giữa vòng đời cùng một ID.
- `coverage_person`, `coverage_bag`: tỷ lệ frame có ít nhất một object class tương ứng.
- `fragment_candidates`: heuristic nối hai track khác ID dựa trên gap, dự đoán chuyển động, IoU và tỷ lệ kích thước.

`fragment_candidates` không phải số ID switch thật. Khi chưa có ground-truth, một tracker có ít track có thể chỉ đang bỏ sót object. Để chọn production tracker, cần gán track ID chuẩn cho một subset và tính HOTA/IDF1/ID switches riêng cho `person` và `bag`.

Chạy unit test metric:

```powershell
.\.venv\Scripts\python.exe -m unittest experiments.tracker_benchmark.test_metrics
```
