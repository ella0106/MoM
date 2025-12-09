from utils.utils import *

from mvextractor.videocap import VideoCap


class MotionVectorExtractor:
    def __init__(self, temp_dir=None, rescale=True, ffmpeg_path=None,
                 max_frames=12, sample_size=6, seed=2024,
                 transcode_height=480,   # 밸런스: 480p(원본이 작으면 스킵)
                 mb_size=16):
        self.temp_dir = temp_dir if temp_dir is not None else 'tmp'
        self.rescale = rescale
        self.ffmpeg_path = ffmpeg_path if ffmpeg_path is not None else '/home/aix21002/.local/bin/ffmpeg'
        self.max_frames = max_frames
        self.fps = sample_size
        self.rng = np.random.default_rng(seed)
        
        self.mean = np.array([[0.0, 0.0]], dtype=np.float64)
        self.std = np.array([[0.0993703, 0.1130276]], dtype=np.float64)
        if self.rescale:
            self.std /= 10.0        
            
        self.transcode_height = transcode_height  # None이면 원본 해상도 유지
        self.mb = mb_size

    def __call__(self, video_path):
        try:
            frames, motion_sequences, motion_indices, motion_raws = self.sample_video_clips(video_path)
            return frames, motion_sequences, motion_indices, motion_raws
        except Exception as e:
            raise ValueError(f"load video motion error {e}")
    
    def extract_motions(self, video_path):
        org_path = video_path
        output_path = os.path.join(self.temp_dir, video_path)
        if os.path.exists(output_path):
            pass
        else:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cmd = [
                self.ffmpeg_path,
                "-loglevel", "error",
                "-i", org_path,
                "-vf", "fps=6,scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",  # 프레임/해상도 정규화
                "-c:v", "libx264",              # H.264 인코딩
                "-preset", "ultrafast",         # 속도 우선
                "-crf", "28",                   # 품질 조정
                "-pix_fmt", "yuv420p",          # 표준 포맷
                "-g", "12",                     # GOP 길이 = 12
                "-keyint_min", "12",            # 최소 I-frame 간격 고정
                "-bf", "0",                     # B-frame 완전 제거
                "-x264-params", "scenecut=0:open_gop=0",  # I-frame 강제 주기화
                "-an",                          # 오디오 제거
                "-y",
                output_path
            ]
            
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = "/home/aix21002/.local/lib:" + env.get("LD_LIBRARY_PATH", "")

            # 실행
            subprocess.run(
                cmd,
                env=env,
                stdout=subprocess.DEVNULL,
                check=True
            )
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
            
            pos = motion_vectors[:, 5:7]
            pos = np.clip(pos, [0, 0], [w - 1, h - 1]).astype(np.int32)
            
            denom = np.maximum(motion_vectors[:, 9:], 1e-6)
            mvs_raw = (motion_vectors[:, 0:1] * motion_vectors[:, 7:9]) / denom  # 픽셀 단위
            mvs_norm = (mvs_raw.astype(np.float32) / np.array([[w, h]], dtype=np.float32) - self.mean) / self.std
            
            mv_norm = np.full((h, w, 2), -10000, dtype=np.float16)
            mv_raw  = np.zeros((h, w, 2), dtype=np.float16)
            y, x = pos[:, 1], pos[:, 0]
            mv_norm[y, x, :] = mvs_norm.astype(np.float16)
            mv_raw[y, x, :]  = mvs_raw.astype(np.float16)

            motions.append((mv_norm, mv_raw))
        return frames, motions, frame_types
                
    def sample_video_clips(self, video_path):
        frames, motions, frame_types = self.extract_motions(video_path)
        p_frames = self.sample_p_indices(frame_types)

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
        # print("motion shapes:", [clips_norm[0].shape])
        return frames, clips_norm, p_frames, clips_raw
   
    def sample_p_indices(self, data):
        sampled_indices = []

        # chunk 단위로 나누기
        for chunk_start in range(0, len(data), self.fps):
            chunk = data[chunk_start:chunk_start + self.fps]
            p_indices = [i for i, val in enumerate(chunk, start=chunk_start)]

            if len(p_indices) < self.fps:
                sampled = sorted(self.rng.choice(p_indices, self.fps, replace=True))
            else:
                sampled = sorted(self.rng.choice(p_indices, self.fps, replace=False))
            sampled_indices.extend(sampled)
            
        data = sampled_indices
        if len(data) % (self.fps*2) != 0:
            pad_len = self.fps*2 - (len(data) % (self.fps*2))
            data = data + [data[-1]] * pad_len
        return data

class MotionVectorProcessor:
    def __init__(self, width=384, height=384, device=None, dtype=torch.float16, chunk=4):
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
            all_motions = [torch.from_numpy(motion) for motion in motions]
        return self.process_in_chunks(all_motions)
    
    def process_in_chunks(self, motions):
        outs = []
        for i in range(0, len(motions), self.chunk):
            chunk_list = motions[i:i+self.chunk]
            part = torch.stack(chunk_list, dim=0).to(self.device, dtype=self.dtype, non_blocking=True)
            outs.append(self.transform_motion(part))
            del part
        torch.cuda.empty_cache()
        return torch.cat(outs, dim=0)
        
    def transform_motion(self, motions):
        with torch.cuda.amp.autocast(enabled=True):
            motions = motions.permute(0, 1, 4, 2, 3).contiguous()  # [B, T, 2, H, W]
            B, T, C, H, W = motions.shape
            
            motions_flat = motions.reshape(B * T, C, H, W)
            padding_h = (16 - H % 16) % 16
            padding_w = (16 - W % 16) % 16

            if padding_h > 0 or padding_w > 0:
                motions_flat = F.pad(motions_flat, (0, padding_w, 0, padding_h), value=-10000)
                
            motions_flat = F.avg_pool2d(motions_flat, kernel_size=16, stride=16)
            motions_flat = motions_flat.masked_fill_(motions_flat < -1e3, 0)
            motions = F.interpolate(motions_flat, size=self.target_size, mode="bilinear", align_corners=False) # [B*T, C, H, W]
            return motions.view(B, T, C, self.target_size[0], self.target_size[1])
        
class ResidualProcessor:
    def __init__(self, frame_num=12, height=384, width=384,
                 device=None, dtype=torch.float16, chunk=4,
                 warp_ratio=1.0):   # 1.0=풀해상도, 0.75=절충, 0.5=스피드
        self.frame_num = frame_num
        self.height = height
        self.width = width
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.chunk = chunk
        self.warp_ratio = warp_ratio

    @torch.no_grad()
    def __call__(self, frames, motions_norm, motion_idx, motions_raw=None):
        if not motions_norm or not motion_idx:
            return torch.empty(0, device=self.device, dtype=self.dtype)

        B = len(motions_raw)
        _, H, W, C = motions_raw[0].shape
        prevs = torch.from_numpy(np.stack([frames[i-1].transpose(2, 0, 1) for i in motion_idx]).reshape(B, -1, 3, H, W)).to(self.dtype)
        currs = torch.from_numpy(np.stack([frames[i].transpose(2, 0, 1) for i in motion_idx]).reshape(B, -1, 3, H, W)).to(self.dtype)
        motions_raw = torch.from_numpy(np.stack(motions_raw).transpose(0, 1, 4, 2, 3)).to(self.dtype)
        
        residuals = []
        for i in range(len(motions_raw)):
            if motion_idx[0] == 0:
                prevs[0][0] = torch.zeros((3, H, W))
            res = self.compute_residual(prevs[i], currs[i], motions_raw[i])
            residuals.append(res.contiguous())
            
        return torch.cat(residuals).view(B, -1, 3, self.height, self.width)
    
    def compute_residual(self, prev, curr, flow):
        """
        prev: [B, 3, H, W]  previous frame
        curr: [B, 3, H, W]  current frame
        flow: [B, 2, H, W]  (dx, dy) in pixel units
        """

        B, C, H, W = prev.shape
        prev = prev.to(self.device)
        curr = curr.to(self.device)
        flow = flow.to(self.device)

        # ===== 1. base grid 생성 =====
        y, x = torch.meshgrid(
            torch.arange(H, device=prev.device),
            torch.arange(W, device=prev.device),
            indexing='ij'
        )
        x = x.float()
        y = y.float()

        # shape: [1,H,W]
        x = x.unsqueeze(0).expand(B, -1, -1)
        y = y.unsqueeze(0).expand(B, -1, -1)

        dx = flow[:, 0]   # [B,H,W]
        dy = flow[:, 1]   # [B,H,W]

        # ===== 2. warp할 목표 좌표 (previous → current) =====
        x2 = x + dx
        y2 = y + dy

        # ===== 3. grid normalize =====
        grid_x = (2.0 * x2 / (W - 1)) - 1.0
        grid_y = (2.0 * y2 / (H - 1)) - 1.0

        grid = torch.stack([grid_x, grid_y], dim=-1).to(self.dtype)  # [B,H,W,2]

        # ===== 4. previous frame warp =====
        warped_prev = F.grid_sample(
            prev, grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        )

        # ===== 5. residual 계산 =====
        residual = curr - warped_prev

        # 영상 코드처럼 128 shifting
        residual = residual + 128.0
        residual = residual.clamp(0, 255)
        residual = F.interpolate(residual, size=(self.height, self.width), mode="bilinear", align_corners=False)

        return residual  # [B,3,H,W]
            
class MotionFeatureExtractor:
    def __init__(self, width=384, height=384, frame_num=12, device=None, dtype=torch.float16):
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
    def __call__(self, frames, motions_norm, motion_indices, motions_raw=None):
        motion_feat = self.motion_processor(motions_norm)                       # [B,T,2,H,W]
        residual_feat = self.residual_processor(frames, motions_norm, motion_indices, motions_raw)  # [B,T,3,H,W]

        rgb_pick = [frames[i] for i in range(0, len(frames), self.frame_num)]
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
    video_path = '/mnt/aix21002/MoM/dataset/Anet/v__-zOtZZ_fwI.mp4'
    import time

    start = time.time() 
    frame_num = 12
    mve = MotionVectorExtractor(temp_dir="MoM/tmp/")
    mfe = MotionFeatureExtractor()

    frames, motions_norm, motion_indices, motions_raw= mve(video_path)   # 프레임, 모션, 인덱스 추출
    end = time.time()
    print(f"Motion Extracting time: {end - start:.6f}초")
    images, motion_feats, residual_feats = mfe(frames, motions_norm, motion_indices, motions_raw)     # RGB + Motion + Residual feature 묶기
    
    print("===== Test Result =====")
    print("RGB feature    :", images.shape)       # [B, H, W]
    print("Motion feature :", motion_feats.shape)    # [B, 2, T, H, W]
    print("Residual feature:", residual_feats.shape) # [B, 3, T, H, W]
    final = time.time() 
    print(f"총 실행 시간: {final - start:.6f}초")