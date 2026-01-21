from utils.utils import *

from mvextractor.videocap import VideoCap


class MotionVectorExtractor:
    def __init__(self, temp_dir=None, rescale=True, ffmpeg_path=None,
                 fps=6, seed=2024):
        self.temp_dir = temp_dir if temp_dir is not None else 'tmp'
        self.rescale = rescale
        self.ffmpeg_path = ffmpeg_path if ffmpeg_path is not None else '/home/aix21002/.local/bin/ffmpeg'
        self.fps = fps
        self.max_frames = fps * 2        
        self.rng = np.random.default_rng(seed)
        
        self.mean = np.array([[0.0, 0.0]], dtype=np.float64)
        self.std = np.array([[0.0993703, 0.1130276]], dtype=np.float64)
        if self.rescale:
            self.std /= 10.0        

    def __call__(self, video_path):
        try:
            frames, motion_sequences, motion_indices, motion_raws = self.sample_video_clips(video_path)
            return frames, motion_sequences, motion_indices, motion_raws
        except Exception as e:
            raise ValueError(f"load video motion error {e}")
    
    def extract_motions(self, video_path):
        org_path = video_path
        filename, filetype = os.path.splitext(video_path)
        output_path = os.path.join(self.temp_dir, f"{filename}_fps{self.fps}{filetype}")
        if os.path.exists(output_path):
            pass
        else:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cmd = [
                self.ffmpeg_path,
                "-loglevel", "error",
                "-i", org_path,
                "-vf", f"fps={self.fps},scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",  # 프레임/해상도 정규화
                "-c:v", "libx264",              # H.264 인코딩
                "-preset", "ultrafast",         # 속도 우선
                "-crf", "28",                   # 품질 조정
                "-pix_fmt", "yuv420p",          # 표준 포맷
                "-g", str(self.fps*2),          # GOP 길이 = 12
                "-keyint_min", str(self.fps*2), # 최소 I-frame 간격 고정
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

        frames, motions_raw, motions_norm, frame_types = [], [], [], []

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
            
            mv_raw  = np.zeros((h, w, 2), dtype=np.float16)
            y, x = pos[:, 1], pos[:, 0]
            mv_raw[y, x, :]  = mvs_raw.astype(np.float16)
            
            mv_norm = self.accumulate_motion_to_coarse(
                pos, mvs_norm, h, w, stride=16
            )

            motions_raw.append(mv_raw)
            motions_norm.append(mv_norm)
        return frames, motions_raw, motions_norm, frame_types
                
    def sample_video_clips(self, video_path):
        frames, motions_raw, motions_norm, frame_types = self.extract_motions(video_path)
        p_frames = self.sample_p_indices(frame_types)

        clips_norm, clips_raw = [], []

        for fid in range(0, len(p_frames), self.max_frames):
            frame_indices = p_frames[fid:fid+self.max_frames]
            
            mn = [motions_norm[i] for i in frame_indices]
            mr = [motions_raw[i] for i in frame_indices]

            clips_norm.append(np.stack(mn, axis=0))  # (T, H, W, 2)
            clips_raw.append(np.stack(mr, axis=0))   # (T, H, W, 2)

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
    
    def accumulate_motion_to_coarse(
        self,
        pos,            # (N,2) int32, (x,y)
        mvs_norm,       # (N,2) float32
        H, W,           # 원해상도
        stride=16,
    ):
        """
        sparse motion → coarse grid (H/stride, W/stride)
        """
        gh, gw = H // stride, W // stride
        mv_coarse = np.zeros((gh, gw, 2), dtype=np.float32)
        cnt = np.zeros((gh, gw), dtype=np.int32)

        gx = pos[:, 0] // stride
        gy = pos[:, 1] // stride

        valid = (gx >= 0) & (gx < gw) & (gy >= 0) & (gy < gh)
        gx, gy = gx[valid], gy[valid]
        vals = mvs_norm[valid]

        np.add.at(mv_coarse, (gy, gx), vals)
        np.add.at(cnt, (gy, gx), 1)

        cnt[cnt == 0] = 1
        mv_coarse /= cnt[..., None]
        return mv_coarse

class MotionVectorProcessor:
    def __init__(self, width=96, height=96, device=None, dtype=torch.float16, chunk=4):
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
        
        m = torch.from_numpy(np.stack(motions)).to(self.device, self.dtype)
        m = m.permute(0,1,4,2,3).contiguous()  # [B,T,2,gh,gw]
        B,T,C,H,W = m.shape
        m = m.view(B*T, C, H, W)
        m = F.interpolate(
            m, size=self.target_size,
            mode="bilinear", align_corners=False
        )
        return m.view(B,T,C,self.target_size[0],self.target_size[1])
    
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
    def __call__(self, frames, motion_idx, motions_raw=None):
        if not motion_idx:
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
    def __init__(self, width=96, height=96, frame_num=12, device=None, dtype=torch.float16, config=None):
        self.motion_processor = MotionVectorProcessor(width=width, height=height, device=device, dtype=dtype)
        self.residual_processor = ResidualProcessor(frame_num=frame_num, height=height, width=width,
                                                    device=device, dtype=dtype)
        self.width = width
        self.height = height
        self.frame_num = frame_num
        self.device = device
        self.dtype = dtype
        self.rgb_size = 384
        self.use_mv = config.use_mv if hasattr(config, 'use_mv') else False
        self.use_residual = config.use_residual if hasattr(config, 'use_residual') else False

    @torch.no_grad()
    def __call__(self, frames, motions_norm, motion_indices, motions_raw=None):
        motion_feat, residual_feat = None, None
        if self.use_mv:
            motion_feat = self.motion_processor(motions_norm)                       # [B,T,2,H,W]
        if self.use_residual:
            residual_feat = self.residual_processor(frames, motion_indices, motions_raw)  # [B,T,3,H,W]

        rgb_pick = [frames[i] for i in range(0, len(frames), self.frame_num)]
        if len(rgb_pick) > 0:
            rgb = np.stack(rgb_pick)                         # [N,H,W,3]
        else:
            rgb = torch.empty(0, device=self.device, dtype=self.dtype)

        return rgb, motion_feat, residual_feat


if __name__ == "__main__":
    video_path = 'dataset/videomme/video/8UxGzDeRIJk.mp4'
    import time

    start = time.time() 
    fps = 4
    mve = MotionVectorExtractor(temp_dir="tmp/", fps=fps)
    mfe = MotionFeatureExtractor(frame_num=fps*2)

    frames, motions_norm, motion_indices, motions_raw= mve(video_path)   # 프레임, 모션, 인덱스 추출
    end = time.time()
    print(f"Motion Extracting time: {end - start:.6f}초")
    images, motion_feats, residual_feats = mfe(frames, motions_norm, motion_indices, motions_raw)     # RGB + Motion + Residual feature 묶기
    
    print("===== Test Result =====")
    print("RGB feature    :", images.shape)       # [B, C, H, W]
    print("Motion feature :", motion_feats.shape)    # [B, 2, T, H, W]
    print("Residual feature:", residual_feats.shape) # [B, 3, T, H, W]
    final = time.time() 
    print(f"총 실행 시간: {final - start:.6f}초")