## 生成全体 秒

| Model | ComfyUI BF16 | rocm BF16 | EXL3 4bit |
|---|---:|---:|---:|
| qwen-base | 50.373 | 35.797 | 39.626 |
| qwen-turbo | 25.062 | 9.156 | 9.835 |
| flux3 | 20.763 | 11.622 | 12.309 |

## Denoising 秒

| Model | ComfyUI BF16 | rocm BF16 | EXL3 4bit |
|---|---:|---:|---:|
| qwen-base | 31.635 | 33.397 | 37.016 |
| qwen-turbo | 6.357 | 6.700 | 7.382 |
| flux3 | 8.143 | 10.014 | 10.710 |

## Peak allocated GiB

| Model | ComfyUI BF16 | rocm BF16 | EXL3 4bit |
|---|---:|---:|---:|
| qwen-base | 30.257 | 15.162 | 5.417 |
| qwen-turbo | 30.257 | 15.162 | 5.417 |
| flux3 | 19.386 | 13.744 | 6.444 |
