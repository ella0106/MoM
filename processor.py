from utils.utils import *

from mvextractor.videocap import VideoCap


class MotionVectorExtractor:
    def __init__(self, temp_dir=None, rescale=True, ffmpeg_path=None,
                 max_frames=12, sample_size=6, seed=2024,
                 transcode_height=480,   # 밸런스: 480p(원본이 작으면 스킵)
                 mb_size=16):
        self.temp_dir = temp_dir if temp_dir is not None else './tmp'
        self.rescale = rescale
        self.ffmpeg_path = ffmpeg_path if ffmpeg_path is not None else '/home/aix23103/anaconda3/envs/llava/bin/ffmpeg'
        self.max_frames = max_frames
        self.sample_size = sample_size
        self.rng = np.random.default_rng(seed)
        
        self.mean = np.array([[0.0, 0.0]], dtype=np.float64)
        self.std = np.array([[0.0993703, 0.1130276]], dtype=np.float64)
        if self.rescale:
            self.std /= 10.0        
            
        self.transcode_height = transcode_height  # None이면 원본 해상도 유지
        self.mb = mb_size

    def __call__(self, video_path):
        try:
            frames, motion_sequences, motion_indices, motion_raws, fps = self.sample_video_clips(video_path)
            return frames, motion_sequences, motion_indices, motion_raws, fps
        except Exception as e:
            raise ValueError(f"load video motion error {e}")
    
    def get_video_fps_cv2(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Failed to open video: {video_path}")
            return None

        # 1️⃣ 기본적으로 메타데이터에서 FPS 읽기
        fps = cap.get(cv2.CAP_PROP_FPS)

        # 2️⃣ FPS 정보가 없으면 직접 계산
        if fps == 0 or fps is None:
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # POS_MSEC은 현재 프레임 위치 기준이므로 끝까지 이동해서 계산
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
            duration_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            if duration_sec > 0:
                fps = frame_count / duration_sec
            else:
                fps = None

        cap.release()
        
        return int(fps)
    
    def extract_motions(self, video_path):
        fps = self.get_video_fps_cv2(video_path)
        output_path = os.path.join(self.temp_dir, f"transcoded_{os.path.basename(video_path)}")
        
        cmd = [
            "ffmpeg",
            "-i", video_path,        # 입력 파일
            "-r", "6",               # 출력 프레임레이트 6fps
            "-c:v", "libx264",       # H.264 인코딩
            "-preset", "medium",     # 인코딩 속도/압축 밸런스
            "-crf", "23",            # 품질 (18~28 권장)
            "-g", "12",              # GOP 길이 = 12
            "-keyint_min", "12",     # 최소 I-frame 간격
            "-bf", "0",              # B-frame 제거
            "-an",                   # 오디오 제거
            "-y",                    # 출력 덮어쓰기
            output_path
        ]

        # 실행
        subprocess.run(cmd, check=True)
        video_path = output_path

        # -----------------------------
        # Motion vector 읽기
        # -----------------------------
        cap = VideoCap()
        ret = cap.open(video_path)
        if not ret:
            raise RuntimeError("VideoCap open 실패 (NAL 손상 가능성)")

        frames, motions, frame_types = [], [], []

        while True:
            ret, frame, motion_vectors, frame_type = cap.read()

            if not ret:
                break
            frames.append(frame[:, :, ::-1])  
            frame_types.append(frame_type)

            h, w = frame.shape[:2]
            mv_norm = np.full((h, w, 2), -10000, dtype=np.float16)
            mv_raw  = np.zeros((h, w, 2), dtype=np.float16)

            # 위치
            position = motion_vectors[:, 5:7].astype(np.int32)
            position[:, 0] = np.clip(position[:, 0], 0, w-1)
            position[:, 1] = np.clip(position[:, 1], 0, h-1)

            denom = np.maximum(motion_vectors[:, 9:], 1e-6)
            mvs_raw = (motion_vectors[:, 0:1] * motion_vectors[:, 7:9]) / denom   # 픽셀 단위

            # 정규화/표준화 버전
            mvs_norm = mvs_raw.copy().astype(np.float16)
            mvs_norm[:, 0] /= float(w)
            mvs_norm[:, 1] /= float(h)
            mvs_norm = (mvs_norm - self.mean) / self.std
            mvs_norm = mvs_norm.astype(np.float16)

            mv_norm[position[:, 1], position[:, 0], :] = mvs_norm
            mv_raw[position[:, 1], position[:, 0], :]  = mvs_raw

            motions.append((mv_norm, mv_raw))   # tuple 저장
        return frames, motions, frame_types, fps
                
    def sample_video_clips(self, video_path):
        frames, motions, frame_types, fps = self.extract_motions(video_path)
        p_frames = self.sample_p_indices(frame_types, fps)

        clips_norm, clips_raw = [], []

        for fid in range(0, len(p_frames), self.max_frames):
            frame_indices = p_frames[fid:fid+self.max_frames]
            mv_norm_list, mv_raw_list = [], []
            
            selected_motions = [motions[i] for i in frame_indices if i < len(motions) and motions[i] is not None]
    
            if selected_motions:
                mv_norm_list, mv_raw_list = zip(*selected_motions)
                
                clip_norm = np.stack(mv_norm_list, axis=0)
                clip_raw = np.stack(mv_raw_list, axis=0)

                # pad if needed
                if clip_norm.shape[0] < self.max_frames:
                    pad_size = self.max_frames - clip_norm.shape[0]
                    pad_norm = np.full((pad_size, *clip_norm.shape[1:]), -10000, dtype=np.float32)
                    pad_raw = np.zeros((pad_size, *clip_raw.shape[1:]), dtype=np.float32)

                    clip_norm = np.concatenate([clip_norm, pad_norm], axis=0)
                    clip_raw = np.concatenate([clip_raw, pad_raw], axis=0)

                clips_norm.append(clip_norm)
                clips_raw.append(clip_raw)

        # print("sampled clips:", len(clips_norm), "raw motions:", len(clips_raw), "frames:", len(frames), "p_frames:", len(p_frames))
        #print("motion shapes:", [clip_norm[0].shape])
        return frames, clips_norm, p_frames, clips_raw, fps
   
    def sample_p_indices(self, data, fps):
        sampled_indices = []

        # chunk 단위로 나누기
        for chunk_start in range(0, len(data), fps):
            chunk = data[chunk_start:chunk_start + fps]
            p_indices = [i for i, val in enumerate(chunk, start=chunk_start) if val == 'P']

            if len(p_indices) < self.sample_size:
                sampled = sorted(self.rng.choice(p_indices, self.sample_size, replace=True))
            else:
                sampled = sorted(self.rng.choice(p_indices, self.sample_size, replace=False))
            sampled_indices.extend(sampled)
            
        data = sampled_indices
        if len(data) % (self.sample_size*2) != 0:
            pad_len = self.sample_size*2 - (len(data) % (self.sample_size*2))
            data = data + [data[-1]] * pad_len
        return data

class MotionVectorProcessor:
    def __init__(self, width=224, height=224, device=None, dtype=torch.float16, chunk=4):
        self.target_size = (height, width)
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # CPU면 FP32 유지
        self.dtype = dtype
        self.chunk = chunk

    @torch.no_grad()
    def __call__(self, motions):
        if not motions:
            return torch.empty(0, device=self.device, dtype=self.dtype)
        if isinstance(motions[0], torch.Tensor):
            all_motions = torch.stack(motions, dim=0)
        else:
            all_motions = torch.from_numpy(np.stack([motion for motion in motions]))
        all_motions = all_motions.to(self.device, dtype=self.dtype, non_blocking=True)
        return self.process_in_chunks(all_motions)
    
    def process_in_chunks(self, motions):
        B = motions.shape[0]
        outs = []
        for i in range(0, B, self.chunk):
            part = motions[i:i+self.chunk]
            outs.append(self.transform_motion(part))
            torch.cuda.empty_cache()
        return torch.cat(outs, dim=0)
        
    def transform_motion(self, motions):
            motions = motions.permute(0, 1, 4, 2, 3).contiguous()  # [B, T, 2, H, W]
            B, T, C, H, W = motions.shape
            
            motions_flat = motions.reshape(B * T, C, H, W)
            padding_h = (16 - H % 16) % 16
            padding_w = (16 - W % 16) % 16

            if padding_h > 0 or padding_w > 0:
                motions_flat = F.pad(motions_flat, (0, padding_w, 0, padding_h), value=-10000)
                
            motions_flat = F.max_pool2d(motions_flat, kernel_size=16, stride=16)
            motions_flat = motions_flat.masked_fill_(motions_flat < -1000, 0)
            motions = F.interpolate(motions_flat, size=self.target_size, mode="bilinear", align_corners=False) # [B*T, C, H, W]
            return motions.view(B, T, C, self.target_size[0], self.target_size[1]).contiguous()
        
class ResidualProcessor:
    def __init__(self, frame_num=12, height=224, width=224,
                 device=None, dtype=torch.float16, chunk=4,
                 warp_ratio=1.0):   # 1.0=풀해상도, 0.75=절충, 0.5=스피드
        self.frame_num = frame_num
        self.height = height
        self.width = width
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.chunk = chunk
        self.warp_ratio = warp_ratio
        
    @functools.lru_cache(maxsize=8)
    def _base_grid(self, H, W):
        # CPU 텐서로 만들어두고 호출 시 device로 옮김
        gy, gx = torch.meshgrid(
            torch.arange(H, dtype=torch.float16),
            torch.arange(W, dtype=torch.float16),
            indexing='ij'
        )
        return torch.stack((gx, gy), dim=2)  # (H,W,2)

    @torch.no_grad()
    def __call__(self, frames, motions_norm, motion_idx, motions_raw=None):
        with torch.no_grad():
            if (not motions_norm) or (not motion_idx):
                return torch.empty(0, device=self.device, dtype=self.dtype)

            prev_frames = torch.from_numpy(np.stack([frames[i - 1] for i in motion_idx]))  # (N,H,W,3)
            N, H0, W0, _ = prev_frames.shape

            # 필요 시 절충 해상도에서 워핑
            Hr = max(1, int(H0 * self.warp_ratio))
            Wr = max(1, int(W0 * self.warp_ratio))

            # 원본 프레임 텐서
            pf = prev_frames.to(device=self.device, dtype=self.dtype) # grid_sample 정밀도↑
            pf = pf.permute(0,3,1,2).contiguous()                                    # (N,3,H,W)
            pf_r = F.interpolate(pf, size=(Hr, Wr), mode="bilinear", align_corners=False)

            # raw flow 준비: [B,T,Hb,Wb,2] -> [N,Hb,Wb,2] -> [N,Hr,Wr,2]
            if motions_raw is None:
                flow_r = torch.zeros((N, Hr, Wr, 2), device=self.device, dtype=torch.float16)
            else:
                raw = torch.from_numpy(np.stack(motions_raw, axis=0)).to(self.device, dtype=torch.float16)  # [B,T,Hb,Wb,2]
                raw = raw.view(-1, raw.shape[2], raw.shape[3], 2)                           # [N,Hb,Wb,2]
                raw = raw.permute(0,3,1,2).contiguous()                                     # [N,2,Hb,Wb]
                flow_r = F.interpolate(raw, size=(Hr, Wr), mode="bilinear", align_corners=False)
                flow_r = flow_r.permute(0,2,3,1).contiguous()                               # [N,Hr,Wr,2]
                # 해상도 바뀌면 플로우 크기도 스케일
                flow_r[..., 0] *= (Wr / float(W0))
                flow_r[..., 1] *= (Hr / float(H0))

            base_grid = self._base_grid(Hr, Wr).to(self.device)        # [Hr,Wr,2]
            grid = base_grid.unsqueeze(0) + flow_r                                          # [N,Hr,Wr,2]
            norm = torch.tensor([2.0/(Wr-1), 2.0/(Hr-1)], device=self.device, dtype=torch.float16)
            grid = grid * norm - 1.0

            # 풀정밀(grid/이미지 float32)로 워핑 → 품질↑, 이후 half로 캐스팅
            warped = F.grid_sample(pf_r, grid, mode='bilinear', padding_mode='border', align_corners=False)
            residuals_r = pf_r - warped
            residuals_r.add_(128.0)
            residuals_r.clamp_(0.0, 255.0)

            # 최종 출력 크기로 리사이즈 + half 캐스팅
            residuals = F.interpolate(residuals_r, size=(self.height, self.width), mode="bilinear", align_corners=False)
            residuals = residuals.to(self.dtype).contiguous()  # [N,3,H,W]

            B = len(motions_norm)
            return residuals.view(B, self.frame_num, 3, self.height, self.width)
            
class MotionFeatureExtractor:
    def __init__(self, width=224, height=224, frame_num=12, device=None, dtype=torch.float16):
        self.motion_processor = MotionVectorProcessor(width=width, height=height, device=device, dtype=dtype)
        self.residual_processor = ResidualProcessor(frame_num=frame_num, height=height, width=width,
                                                    device=device, dtype=dtype)
        self.width = width
        self.height = height
        self.frame_num = frame_num
        self.device = device
        self.dtype = dtype
        self.rgb_size = 384

    @torch.no_grad()
    def __call__(self, frames, motions_norm, motion_indices, motions_raw=None, fps=None):
        motion_feat = self.motion_processor(motions_norm)                       # [B,T,2,H,W]
        residual_feat = self.residual_processor(frames, motions_norm, motion_indices, motions_raw)  # [B,T,3,H,W]

        rgb_pick = [frames[i] for i in range(0, len(frames), fps*2)]
        if len(rgb_pick) > 0:
            rgb_np = np.stack(rgb_pick)                         # [N,H,W,3]
            rgb = torch.from_numpy(rgb_np).to(self.device, dtype=torch.float32)
            rgb = rgb.permute(0,3,1,2).contiguous()             # [N,3,H,W]
            rgb = F.interpolate(rgb, size=(self.rgb_size, self.rgb_size),
                                mode="bilinear", align_corners=False)  # [N,3,384,384]
            rgb = rgb.to(self.dtype).contiguous()
        else:
            rgb = torch.empty(0, device=self.device, dtype=self.dtype)

        return rgb, motion_feat, residual_feat


if __name__ == "__main__":
    video_path = '/data2/aix23103/LLaVA-NeXT/cat_and_chicken.mp4'
    import time

    start = time.time() 
    frame_num = 12
    mve = MotionVectorExtractor()
    mfe = MotionFeatureExtractor()

    frames, motions_norm, motion_indices, motions_raw, fps= mve(video_path)   # 프레임, 모션, 인덱스 추출
    features = mfe(frames, motions_norm, motion_indices, motions_raw, fps)     # RGB + Motion + Residual feature 묶기
    end = time.time() 
    
    print("===== Test Result =====")
    print("RGB feature    :", features["rgb"].shape)       # [H, W, T]
    print("Motion feature :", features["motion"].shape)    # [B, 2, T, H, W]
    print("Residual feature:", features["residual"].shape) # [B, 3, T, H, W]
    print(f"실행 시간: {end - start:.6f}초")